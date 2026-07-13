"""
Topic 44: RAGAS & Evaluating RAG Pipelines
===========================================
RAGAS (Retrieval Augmented Generation Assessment) scores a RAG pipeline on
four independent axes instead of one opaque "accuracy" number. The real
`ragas` package calls an LLM judge for every score, which needs an API key.
This file reimplements all four metrics from scratch using simple word/
number overlap heuristics — no external dependencies, fully offline.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field

DIVIDER = "=" * 70
THIN = "-" * 70
WRAP_W = 66

# Threshold below which a metric is considered "weak" for diagnosis.
WEAK_THRESHOLD = 0.5


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=WRAP_W, initial_indent=indent,
                          subsequent_indent=indent)


# ---------------------------------------------------------------------------
# TOKENIZER — shared bag-of-words helper used by every metric below
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "as", "at", "by", "be", "this", "that", "are", "with", "its",
    "was", "can", "more", "from", "into", "such", "both", "their", "also",
    "they", "have", "has", "been", "does", "do", "did", "you", "your",
    "we", "our", "i", "about", "than", "so", "not", "but", "if", "then",
}


def _tokenize(text: str) -> set[str]:
    """
    Lower-case a string and return the set of "key terms": content words
    (with stopwords removed) plus any standalone numbers. Numbers are kept
    because factual claims ("2007", "100 degrees") hinge on them just as
    much as nouns do.
    """
    raw = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())
    return {tok for tok in raw if tok not in _STOPWORDS and len(tok) > 1}


def _overlap_fraction(query_tokens: set[str], target_tokens: set[str]) -> float:
    """Fraction of query_tokens that also appear in target_tokens."""
    if not query_tokens:
        return 1.0
    return len(query_tokens & target_tokens) / len(query_tokens)


# ---------------------------------------------------------------------------
# THE FOUR RAGAS METRICS — reimplemented with overlap heuristics
# ---------------------------------------------------------------------------

def faithfulness(answer: str, retrieved_contexts: list[str]) -> float:
    """
    "Did the model stick to what it retrieved?"

    Real RAGAS decomposes the answer into atomic claims and asks an LLM
    judge whether each is entailed by the context. We approximate a
    "claim" with the answer's key terms and check what fraction appear
    somewhere in the retrieved contexts — terms the generator invented
    that never showed up in retrieval count as hallucinations.
    """
    answer_terms = _tokenize(answer)
    context_terms: set[str] = set()
    for chunk in retrieved_contexts:
        context_terms |= _tokenize(chunk)
    return round(_overlap_fraction(answer_terms, context_terms), 3)


def answer_relevance(question: str, answer: str) -> float:
    """
    "Did the model actually answer the question that was asked?"

    Real RAGAS generates reverse-questions from the answer and compares
    their embeddings to the original question. We approximate this with
    how many of the question's key terms are echoed back in the answer —
    an answer faithful to context but off-topic scores low here even
    when faithfulness looks fine.
    """
    question_terms = _tokenize(question)
    answer_terms = _tokenize(answer)
    return round(_overlap_fraction(question_terms, answer_terms), 3)


def context_precision(question: str, retrieved_contexts: list[str],
                       relevance_threshold: float = 0.2) -> float:
    """
    "Of everything retrieved, how much of it was actually useful?"

    Real RAGAS asks an LLM to judge each chunk against the question. We
    approximate that with word overlap: a chunk is "relevant" once its
    question-term coverage exceeds `relevance_threshold`. Precision is
    the fraction of chunks that clear the bar — noisy chunks drag the
    score down even if one good chunk is buried in the pile.
    """
    if not retrieved_contexts:
        return 0.0
    question_terms = _tokenize(question)
    relevant = 0
    for chunk in retrieved_contexts:
        chunk_terms = _tokenize(chunk)
        overlap = _overlap_fraction(question_terms, chunk_terms)
        if overlap >= relevance_threshold:
            relevant += 1
    return round(relevant / len(retrieved_contexts), 3)


def context_recall(retrieved_contexts: list[str], reference_answer: str) -> float:
    """
    "Did retrieval bring back everything needed to answer correctly?"

    Real RAGAS decomposes the reference answer into atomic statements and
    checks each against the retrieved context. We approximate a
    "statement" with the reference answer's key terms and measure what
    fraction are present *somewhere* in the retrieved chunks — independent
    of what the generator actually wrote, isolating retrieval quality.
    """
    reference_terms = _tokenize(reference_answer)
    context_terms: set[str] = set()
    for chunk in retrieved_contexts:
        context_terms |= _tokenize(chunk)
    return round(_overlap_fraction(reference_terms, context_terms), 3)


# ---------------------------------------------------------------------------
# EVALUATION SET — 5 (question, retrieved_contexts, answer, reference) rows
# ---------------------------------------------------------------------------
# Each row is engineered to exercise a different failure mode so the demo
# clearly shows which pipeline stage — retrieval or generation — is broken.

@dataclass
class EvalCase:
    label: str
    question: str
    retrieved_contexts: list[str]
    generated_answer: str
    reference_answer: str
    scores: dict = field(default_factory=dict)


EVAL_SET: list[EvalCase] = [
    EvalCase(
        label="Healthy pipeline",
        question="Who invented the World Wide Web and in what year?",
        retrieved_contexts=[
            "Tim Berners-Lee invented the World Wide Web in 1989 while "
            "working at CERN in Switzerland.",
            "The web relies on the HTTP protocol to transfer hypertext "
            "documents between clients and servers.",
        ],
        generated_answer=(
            "Tim Berners-Lee invented the World Wide Web in 1989 while "
            "working at CERN."
        ),
        reference_answer=(
            "The World Wide Web was invented by Tim Berners-Lee in 1989 "
            "at CERN."
        ),
    ),
    EvalCase(
        label="Hallucinated detail (generation problem)",
        question="What year was the first iPhone released?",
        retrieved_contexts=[
            "Apple released the first iPhone in 2007.",
            "The iPhone popularized capacitive touchscreens on smartphones.",
        ],
        generated_answer=(
            "The first iPhone was released in 2007 and sold over 10 "
            "million units in its opening weekend, making it an instant "
            "record-breaking blockbuster."
        ),
        reference_answer="The first iPhone was released by Apple in 2007.",
    ),
    EvalCase(
        label="Retrieval missed key facts (recall problem)",
        question="What are the main components of a neural network?",
        retrieved_contexts=[
            "A neural network consists of layers of interconnected nodes "
            "called neurons.",
        ],
        generated_answer=(
            "The main components of a neural network are layers of "
            "interconnected neurons."
        ),
        reference_answer=(
            "A neural network consists of neurons organized in layers, "
            "connected by weighted edges, with activation functions that "
            "introduce non-linearity, and trained via backpropagation."
        ),
    ),
    EvalCase(
        label="On-topic context, off-topic answer (relevance problem)",
        question="What is the boiling point of water at sea level?",
        retrieved_contexts=[
            "Water boils at 100 degrees Celsius at standard atmospheric "
            "pressure at sea level.",
            "Boiling point decreases with altitude because atmospheric "
            "pressure drops.",
        ],
        generated_answer=(
            "Boiling point decreases with altitude because atmospheric "
            "pressure drops as elevation increases."
        ),
        reference_answer=(
            "Water boils at 100 degrees Celsius at sea level."
        ),
    ),
]


# ---------------------------------------------------------------------------
# SCORING + DIAGNOSIS
# ---------------------------------------------------------------------------

def score_case(case: EvalCase) -> dict:
    scores = {
        "faithfulness": faithfulness(case.generated_answer, case.retrieved_contexts),
        "answer_relevance": answer_relevance(case.question, case.generated_answer),
        "context_precision": context_precision(case.question, case.retrieved_contexts),
        "context_recall": context_recall(case.retrieved_contexts, case.reference_answer),
    }
    case.scores = scores
    return scores


def diagnose(scores: dict) -> str:
    """
    Turn four numbers into an actionable verdict: is the weak link
    retrieval (bad context_precision / context_recall) or generation
    (low faithfulness or answer_relevance despite decent retrieval)?
    """
    weak_precision = scores["context_precision"] < WEAK_THRESHOLD
    weak_recall = scores["context_recall"] < WEAK_THRESHOLD
    weak_faithfulness = scores["faithfulness"] < WEAK_THRESHOLD
    weak_relevance = scores["answer_relevance"] < WEAK_THRESHOLD

    retrieval_ok = not weak_precision and not weak_recall

    if weak_faithfulness and retrieval_ok:
        return "Generation — model hallucinated beyond good context"
    if weak_precision and weak_recall:
        return "Retrieval — noisy AND incomplete context"
    if weak_precision:
        return "Retrieval — too many irrelevant chunks (low precision)"
    if weak_recall:
        return "Retrieval — missing needed facts (low recall)"
    if weak_relevance:
        return "Generation — answer drifted off-topic"
    return "Healthy — all four metrics look good"


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------

def truncate(text: str, width: int = 34) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def print_results_table(cases: list[EvalCase]) -> None:
    col_q, col_m = 34, 8
    header = (
        f"  {'Question':<{col_q}} {'Faith':<{col_m}} {'AnsRel':<{col_m}} "
        f"{'CtxPrec':<{col_m}} {'CtxRec':<{col_m}}"
    )
    print(header)
    print("  " + "-" * (col_q + 4 * col_m + 4))
    for case in cases:
        s = case.scores
        print(
            f"  {truncate(case.question, col_q):<{col_q}} "
            f"{s['faithfulness']:<{col_m}} {s['answer_relevance']:<{col_m}} "
            f"{s['context_precision']:<{col_m}} {s['context_recall']:<{col_m}}"
        )


def print_case_detail(i: int, case: EvalCase) -> None:
    print(DIVIDER)
    print(f"  CASE {i}: {case.label}")
    print(DIVIDER)
    print(f"  Question: {case.question}")
    print()
    print("  Retrieved contexts:")
    for j, chunk in enumerate(case.retrieved_contexts, start=1):
        print(f"    [{j}] {chunk}")
    print()
    print("  Generated answer:")
    print(wrap(case.generated_answer, indent="    "))
    print()
    print("  Reference (ground truth) answer:")
    print(wrap(case.reference_answer, indent="    "))
    print()
    s = case.scores
    print("  Scores")
    print(THIN)
    metric_notes = [
        ("faithfulness", "Faithfulness", "claims in answer supported by context"),
        ("answer_relevance", "Answer Relevance", "does the answer address the question?"),
        ("context_precision", "Context Precision", "fraction of retrieved chunks that are on-topic"),
        ("context_recall", "Context Recall", "fraction of needed facts that were retrieved"),
    ]
    for key, name, note in metric_notes:
        print(f"    {name:<19}{s[key]:.3f}   ({note})")
    print()
    print(f"  DIAGNOSIS: {diagnose(s)}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 44: RAGAS & EVALUATING RAG PIPELINES")
    print(DIVIDER)
    print(
        "\n  RAGAS scores a RAG pipeline on four independent axes instead\n"
        "  of one end-to-end accuracy number:\n"
        "\n"
        "    1. Faithfulness       — is the answer grounded in context?\n"
        "    2. Answer Relevance   — does the answer address the question?\n"
        "    3. Context Precision  — is retrieval free of irrelevant noise?\n"
        "    4. Context Recall     — did retrieval find everything needed?\n"
        "\n"
        "  All four are computed here with word/number overlap heuristics\n"
        "  — the real `ragas` package uses an LLM judge instead."
    )
    print()

    for case in EVAL_SET:
        score_case(case)

    for i, case in enumerate(EVAL_SET, start=1):
        print_case_detail(i, case)

    print(DIVIDER)
    print("  SUMMARY RESULTS TABLE")
    print(DIVIDER)
    print_results_table(EVAL_SET)
    print()

    print(DIVIDER)
    print("  DIAGNOSIS SUMMARY — which stage needs fixing?")
    print(DIVIDER)
    for i, case in enumerate(EVAL_SET, start=1):
        print(f"  {i}. {truncate(case.question, 50):<50} -> {diagnose(case.scores)}")
    print()

    print(DIVIDER)
    print("  WHY FOUR METRICS INSTEAD OF ONE SCORE?")
    print(DIVIDER)
    print(
        "\n  A single end-to-end accuracy number says only that something\n"
        "  is wrong — never which piece to fix. RAG has two independently\n"
        "  failing stages: RETRIEVAL (did we find the right documents?)\n"
        "  and GENERATION (did the model use them correctly?). Collapsing\n"
        "  both into one score means every regression looks the same, so\n"
        "  every fix is a guess.\n"
        "\n"
        "  Interpreting each metric in practice:\n"
        "    - Low faithfulness      -> GENERATION fix: tighten grounding\n"
        "                               instructions, add citations.\n"
        "    - Low answer relevance  -> GENERATION fix: the model ignored\n"
        "                               the question; fix the prompt.\n"
        "    - Low context precision -> RETRIEVAL fix: re-rank results or\n"
        "                               shrink chunk size to cut noise.\n"
        "    - Low context recall    -> RETRIEVAL fix: raise top-k or\n"
        "                               improve chunking to stop splitting\n"
        "                               facts across chunk boundaries.\n"
        "\n"
        "  Case 2 has near-perfect retrieval but a hallucinated detail —\n"
        "  purely a generation bug. Case 3 has a faithful, on-topic answer\n"
        "  that is simply incomplete — purely a retrieval bug. One accuracy\n"
        "  score would flag both as 'wrong' and tell nobody which team\n"
        "  should fix it."
    )
    print()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. RAGAS decomposes 'is my RAG good?' into four testable axes:",
        "   faithfulness, answer relevance, context precision, recall.",
        "2. Faithfulness catches hallucination — claims not backed by the",
        "   retrieved context, even if the retrieved context was correct.",
        "3. Context precision and recall isolate retrieval quality, fully",
        "   independent of what the generator ultimately wrote.",
        "4. Only context recall needs a ground-truth reference answer; the",
        "   other three metrics work on live traffic with no labels.",
        "5. Low faithfulness with healthy retrieval = fix the prompt/model.",
        "   Low precision/recall = fix the retriever, not the generator.",
        "6. This entire evaluation harness ran with zero external",
        "   dependencies and no LLM API calls.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
