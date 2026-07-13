"""
Topic 42: Eval Methods — Exact Match, LLM-as-Judge, Rubrics
=============================================================
Builds a minimal LLM-as-judge pipeline over a set of question/answer pairs
and compares it against two simpler evaluation methods: exact match and
keyword-overlap (F1) scoring. All three are implemented from scratch with
zero external dependencies and run fully offline.
"""

from __future__ import annotations

import re
import string
import textwrap

DIVIDER = "=" * 70
THIN = "-" * 70


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=66, initial_indent=indent, subsequent_indent=indent)


# ---------------------------------------------------------------------------
# 1. DATASET — question / model answer / reference answer triples
# ---------------------------------------------------------------------------
# Deliberately varied: some model answers are correct-but-differently-worded
# paraphrases, one is a straight factual error, one is verbose-and-off-topic,
# one is a "negation trap" (shares vocabulary with the reference but flips
# the meaning), and one is only partially correct. This mix is what makes
# the three eval methods disagree with each other below.

QA_PAIRS = [
    {"question": "What is the capital of France?",
     "reference_answer": "Paris",
     "model_answer": "The capital of France is Paris."},
    {"question": "Who wrote the play 'Romeo and Juliet'?",
     "reference_answer": "William Shakespeare",
     "model_answer": "William Shakespeare wrote it in the late 16th century."},
    {"question": "What is the boiling point of water at sea level?",
     "reference_answer": "100 degrees Celsius",
     "model_answer": "Water boils at 100°C when measured at sea level."},
    {"question": "What is the largest planet in the solar system?",
     "reference_answer": "Jupiter",
     "model_answer": "Saturn is the largest planet in the solar system."},
    {"question": "What is the chemical symbol for water?",
     "reference_answer": "H2O",
     "model_answer": "H2O"},
    {"question": "In what year did World War II end?",
     "reference_answer": "1945",
     "model_answer": "The Great Depression caused significant economic "
                      "hardship worldwide during the 1930s."},
    {"question": "Does drinking coffee stunt growth in children?",
     "reference_answer": "No, this is a myth; there is no scientific evidence "
                          "that coffee stunts growth in children.",
     "model_answer": "Yes, coffee stunts growth in children because caffeine "
                      "affects bone density."},
    {"question": "What are the primary colors?",
     "reference_answer": "Red, blue, and yellow.",
     "model_answer": "Red and blue are primary colors."},
]


# ---------------------------------------------------------------------------
# 2. METHOD 1 — EXACT MATCH
# ---------------------------------------------------------------------------
# Strict normalized string equality. Cheap and deterministic, but brittle:
# any rewording, extra words, or different formatting causes a false "WRONG"
# even when the model answer is fully correct.

def _normalize_strict(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    lowered = text.lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    return " ".join(no_punct.split())


def exact_match(model_answer: str, reference_answer: str) -> bool:
    return _normalize_strict(model_answer) == _normalize_strict(reference_answer)


# ---------------------------------------------------------------------------
# 3. METHOD 2 — KEYWORD OVERLAP (PRECISION / RECALL / F1)
# ---------------------------------------------------------------------------
# More lenient than exact match, but still surface-level: it has no notion
# of meaning, so it can be fooled by shared words even when the underlying
# claim is reversed (see the "coffee stunts growth" example below).

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "it", "its", "in", "on",
    "of", "to", "and", "or", "for", "as", "at", "by", "be", "this", "that",
    "with", "from", "when", "because", "due", "there", "who", "what",
    "does", "do", "did", "has", "have", "been", "during", "no", "yes",
}


def _content_words(text: str) -> set[str]:
    """Lower-case alphabetic tokens, length > 1, with stop words removed."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in _STOPWORDS}


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+\.?\d*", text))


def _precision_recall_f1(hyp_tokens: set[str], ref_tokens: set[str]) -> tuple[float, float, float]:
    """Precision/recall/F1 of hyp_tokens against ref_tokens (token-set based)."""
    if not hyp_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    common = hyp_tokens & ref_tokens
    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def keyword_overlap_score(model_answer: str, reference_answer: str) -> float:
    """F1 overlap between the model answer's and reference answer's word sets."""
    hyp_tokens = _content_words(model_answer)
    ref_tokens = _content_words(reference_answer)
    _, _, f1 = _precision_recall_f1(hyp_tokens, ref_tokens)
    return f1


# ---------------------------------------------------------------------------
# 4. METHOD 3 — LLM-AS-JUDGE
# ---------------------------------------------------------------------------
# In production this is a real LLM call with a structured rubric prompt: the
# model reads the question, the reference answer, and the candidate answer,
# then returns a score + justification. No API key is available offline, so
# `llm_as_judge()` below is a deterministic STAND-IN that inspects the same
# kind of signals a real judge would (concept coverage, key facts/numbers,
# contradictions) — approximating the same reasoning without a real call.

RUBRIC = {
    "accuracy": "1 = completely wrong or contradicts the reference; "
                "5 = fully correct, matches the reference's key facts",
    "completeness": "1 = misses essentially all reference information; "
                     "5 = covers everything the reference answer covers",
}

# The prompt a REAL LLM-as-judge call would receive in production:
JUDGE_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an evaluation judge. Score the model's answer on a 1-5 scale
    for each criterion below, and give a one-line justification.

    Question: {question}
    Reference answer: {reference_answer}
    Model answer: {model_answer}

    Criteria:
    - Accuracy: {accuracy_rubric}
    - Completeness: {completeness_rubric}

    Respond as JSON: {{"accuracy": N, "completeness": N, "justification": "..."}}
""")


def _detect_contradiction(model_answer: str, reference_answer: str) -> bool:
    """Flag a leading yes/no mismatch — a cheap proxy for a reversed claim."""
    m, r = model_answer.strip().lower(), reference_answer.strip().lower()
    return (m.startswith("yes") and r.startswith("no")) or (m.startswith("no") and r.startswith("yes"))


def llm_as_judge(question: str, model_answer: str, reference_answer: str, rubric: dict) -> dict:
    """
    Deterministic stand-in for a real LLM-as-judge call. Combines keyword/
    concept coverage of the reference answer, presence (or absence) of key
    numbers from the reference, and a contradiction check (reversed yes/no
    claims) into an Accuracy score, a Completeness score, and a one-line
    justification.
    """
    del rubric  # not used by scoring, only shown in the prompt template above

    ref_tokens, model_tokens = _content_words(reference_answer), _content_words(model_answer)
    ref_numbers, model_numbers = _extract_numbers(reference_answer), _extract_numbers(model_answer)
    overlap = ref_tokens & model_tokens
    word_cov = (len(overlap) / len(ref_tokens)) if ref_tokens else None
    num_cov = (1.0 if (ref_numbers & model_numbers) else 0.0) if ref_numbers else None
    coverage = word_cov if word_cov is not None else 0.0
    contradiction = _detect_contradiction(model_answer, reference_answer)
    signals = []

    # --- Accuracy: is what's asserted correct? ----------------------------
    if contradiction:
        accuracy = 1
        signals.append("model directly contradicts the reference (yes/no mismatch)")
    elif ref_numbers:
        if ref_numbers & model_numbers:
            accuracy = 5
            signals.append(f"key figure(s) {sorted(ref_numbers & model_numbers)} confirmed")
        else:
            accuracy = 1 if coverage == 0 else 2
            signals.append(f"reference figure(s) {sorted(ref_numbers)} missing or contradicted")
    elif coverage >= 0.6:
        accuracy = 5
        signals.append(f"covers {len(overlap)}/{len(ref_tokens)} key reference concept(s)")
    elif coverage >= 0.34:
        accuracy = 4
        signals.append(f"covers {len(overlap)}/{len(ref_tokens)} key reference concept(s)")
    elif coverage > 0:
        accuracy = 2
        signals.append(f"only {len(overlap)}/{len(ref_tokens)} key reference concept(s) present")
    else:
        accuracy = 1
        signals.append("shares no key concepts with the reference answer")

    # --- Completeness: how much of the reference's info is reflected? -----
    parts = [c for c in (word_cov, num_cov) if c is not None]
    combined_cov = sum(parts) / len(parts) if parts else 0.0
    if combined_cov >= 0.8:
        completeness = 5
    elif combined_cov >= 0.6:
        completeness = 4
    elif combined_cov >= 0.3:
        completeness = 3
    elif combined_cov > 0:
        completeness = 2
    else:
        completeness = 1

    missing = ref_tokens - model_tokens
    if missing and completeness < 5:
        signals.append(f"missing concept(s): {', '.join(sorted(missing))}")

    return {"accuracy": accuracy, "completeness": completeness, "justification": "; ".join(signals)}


# ---------------------------------------------------------------------------
# 5. RUN ALL THREE METHODS OVER THE DATASET
# ---------------------------------------------------------------------------

def evaluate_all(qa_pairs: list[dict]) -> list[dict]:
    results = []
    for pair in qa_pairs:
        q, model_a, ref_a = pair["question"], pair["model_answer"], pair["reference_answer"]
        judge = llm_as_judge(q, model_a, ref_a, RUBRIC)
        results.append({
            "question": q, "model_answer": model_a, "reference_answer": ref_a,
            "exact_match": exact_match(model_a, ref_a),
            "keyword_f1": keyword_overlap_score(model_a, ref_a),
            "judge_accuracy": judge["accuracy"],
            "judge_completeness": judge["completeness"],
            "judge_justification": judge["justification"],
        })
    return results


# ---------------------------------------------------------------------------
# 6. DISPLAY — RESULTS TABLE
# ---------------------------------------------------------------------------

def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def print_results_table(results: list[dict]) -> None:
    q_w, em_w, kf_w, ja_w, jc_w, just_w = 32, 6, 6, 5, 5, 34
    print(f"  {'Question':<{q_w}} {'Exact':<{em_w}} {'KwF1':<{kf_w}} "
          f"{'JAcc':<{ja_w}} {'JCmp':<{jc_w}} Judge Justification")
    print("  " + "-" * (q_w + em_w + kf_w + ja_w + jc_w + just_w))
    for r in results:
        q = _truncate(r["question"], q_w)
        em = "MATCH" if r["exact_match"] else "WRONG"
        just = _truncate(r["judge_justification"], just_w)
        print(f"  {q:<{q_w}} {em:<{em_w}} {r['keyword_f1']:<{kf_w}.2f} "
              f"{r['judge_accuracy']:<{ja_w}} {r['judge_completeness']:<{jc_w}} {just}")
    print()


def print_full_justifications(results: list[dict]) -> None:
    print("  Full judge justifications")
    print(THIN)
    for i, r in enumerate(results, start=1):
        print(f"  {i}. {r['question']}")
        print(wrap(f"Judge: {r['judge_justification']}", indent="     "))
    print()


# ---------------------------------------------------------------------------
# 7. WHERE THE METHODS DISAGREE
# ---------------------------------------------------------------------------

def print_disagreements(results: list[dict]) -> None:
    print(DIVIDER)
    print("  WHERE THE METHODS DISAGREE")
    print(DIVIDER)
    print(wrap("The point of running three methods side by side is to see where "
               "they disagree — that's exactly where a single metric would mislead you."))
    print()

    for r in results:
        if not r["exact_match"] and r["judge_accuracy"] >= 4 and r["keyword_f1"] < 0.2:
            print(f"  * \"{r['question']}\"")
            print(wrap(f"Reference: '{r['reference_answer']}'  |  Model: '{r['model_answer']}'",
                       indent="      "))
            print(wrap(
                f"Exact match says WRONG (strings differ). Keyword F1 is only "
                f"{r['keyword_f1']:.2f} (almost no shared vocabulary — 'Celsius' vs '°C' "
                f"never overlap as tokens). Yet the judge correctly scores this "
                f"{r['judge_accuracy']}/5 accurate, because it grounds the check on the key "
                f"FACT (the number 100) rather than surface wording. This is the core lesson "
                f"of LLM-as-judge: it can recognize a valid paraphrase that both cheaper "
                f"methods miss.", indent="      "))
            print()

        if r["keyword_f1"] >= 0.4 and r["judge_accuracy"] <= 2:
            print(f"  * \"{r['question']}\"")
            print(wrap(f"Reference: '{r['reference_answer']}'  |  Model: '{r['model_answer']}'",
                       indent="      "))
            print(wrap(
                f"Keyword F1 is a deceptively high {r['keyword_f1']:.2f} because the model "
                f"reused most of the reference's vocabulary (coffee, stunts, growth, "
                f"children) — but it flipped the claim entirely (yes vs no). Keyword overlap "
                f"has no way to notice that. The judge catches it and scores it "
                f"{r['judge_accuracy']}/5, matching exact match's WRONG verdict for the right "
                f"reason.", indent="      "))
            print()


# ---------------------------------------------------------------------------
# 8. METHOD COMPARISON
# ---------------------------------------------------------------------------

def print_method_comparison() -> None:
    print(DIVIDER)
    print("  METHOD COMPARISON: EXACT MATCH vs KEYWORD OVERLAP vs LLM-AS-JUDGE")
    print(DIVIDER)

    rows = [
        ("Cost", "Free", "Free", "$ per call (real LLM)"),
        ("Speed", "Instant", "Instant", "Seconds per call (real LLM)"),
        ("Handles paraphrasing", "No", "Partially (shared words only)", "Yes — reasons about meaning"),
        ("Needs reference answer", "Yes", "Yes", "Optional (rubric alone can work)"),
        ("Reliability concerns", "None (deterministic)", "Fooled by shared vocab / negation",
         "Judge bias: position, verbosity, self-enhancement"),
        ("Best for", "Closed-form factual QA", "Summarization overlap (ROUGE/BLEU-like)",
         "Open-ended, nuanced, subjective tasks"),
    ]
    c1, c2, c3, c4 = 24, 24, 30, 30
    print(f"  {'Aspect':<{c1}} {'Exact Match':<{c2}} {'Keyword Overlap':<{c3}} {'LLM-as-Judge':<{c4}}")
    print("  " + "-" * (c1 + c2 + c3 + c4))
    for aspect, em, kw, judge in rows:
        em_l, kw_l, judge_l = textwrap.wrap(em, c2) or [""], textwrap.wrap(kw, c3) or [""], textwrap.wrap(judge, c4) or [""]
        for i in range(max(len(em_l), len(kw_l), len(judge_l))):
            a = aspect if i == 0 else ""
            e = em_l[i] if i < len(em_l) else ""
            k = kw_l[i] if i < len(kw_l) else ""
            j = judge_l[i] if i < len(judge_l) else ""
            print(f"  {a:<{c1}} {e:<{c2}} {k:<{c3}} {j:<{c4}}")
    print()
    print(wrap("Guidance: use exact match for closed-form factual answers with one correct "
               "string (or a short whitelist). Use keyword/ROUGE-style overlap for tasks like "
               "summarization where surface-level recall is a reasonable proxy. Use "
               "LLM-as-judge with a structured rubric for open-ended, creative, or subjective "
               "answers — but mitigate its biases by swapping answer order, instructing it to "
               "ignore length, and spot-checking with human review."))
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 42: EVAL METHODS — EXACT MATCH, LLM-AS-JUDGE, RUBRICS")
    print(DIVIDER)
    print()
    print(wrap("Grading AI answers is like grading a math test vs an essay: exact match "
               "works for single-right-answer questions, but open-ended answers need richer "
               "methods. This demo runs the SAME question/answer pairs through three eval "
               "methods and shows exactly where they disagree."))
    print()
    print(f"  Dataset size: {len(QA_PAIRS)} question/answer pairs")
    print()

    print(THIN)
    print("  The rubric-driven prompt a REAL LLM-as-judge call would send:")
    print(THIN)
    example_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=QA_PAIRS[0]["question"],
        reference_answer=QA_PAIRS[0]["reference_answer"],
        model_answer=QA_PAIRS[0]["model_answer"],
        accuracy_rubric=RUBRIC["accuracy"],
        completeness_rubric=RUBRIC["completeness"],
    )
    for line in example_prompt.splitlines():
        print(f"  {line}")
    print(wrap("No API key is available offline, so llm_as_judge() below simulates the same "
               "reasoning deterministically (concept coverage, key-fact/number grounding, "
               "contradiction detection) instead of making a real call."))
    print()

    results = evaluate_all(QA_PAIRS)

    print(DIVIDER)
    print("  RESULTS TABLE")
    print(DIVIDER)
    print_results_table(results)
    print_full_justifications(results)
    print_disagreements(results)
    print_method_comparison()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. Exact match is cheap and deterministic but brittle — a correct paraphrase",
        "   ('100°C' vs '100 degrees Celsius') fails it outright.",
        "2. Keyword/F1 overlap is more lenient but still purely lexical — it can be",
        "   fooled by shared vocabulary even when the claim is reversed (negation).",
        "3. LLM-as-judge combines multiple signals (fact grounding, contradiction",
        "   checks, concept coverage) to approximate real semantic understanding.",
        "4. Accuracy and completeness are separate axes: an answer can be fully",
        "   accurate but incomplete, or vice versa — score them independently.",
        "5. In production, replace the deterministic stand-in with a real LLM call",
        "   using a structured rubric prompt — but mitigate position/verbosity bias.",
        "6. No single method wins universally: match the eval method to the task.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
