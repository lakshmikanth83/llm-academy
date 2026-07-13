"""
Topic 61: Mini End-to-End RAG + Agent Demo
==========================================
A minimal agent loop that combines retrieval-augmented generation with
simple decision-making — all offline, no external dependencies.

Architecture
------------
  Corpus       5 short paragraphs on AI topics (in-memory)
  Retriever    Jaccard word-overlap similarity
  Agent        Decides each turn whether to:
               a) Retrieve docs and answer with context  (RAG path)
               b) Answer from a small built-in knowledge dict (direct path)
  Decision     Overlap between query tokens and corpus tokens drives routing:
               if best Jaccard score >= ROUTE_THRESHOLD → RAG path
               otherwise                                → direct path
  Responses    Simulated (template-based) — no API calls required
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOP_K           = 2      # number of documents to retrieve
ROUTE_THRESHOLD = 0.06   # Jaccard score above which agent chooses RAG path


# ---------------------------------------------------------------------------
# 1. CORPUS  (5 paragraphs about AI)
# ---------------------------------------------------------------------------

CORPUS: list[dict] = [
    {
        "id":    "C1",
        "title": "What is machine learning?",
        "text": (
            "Machine learning is a branch of artificial intelligence in which "
            "systems learn from data to improve their performance on a task without "
            "being explicitly programmed. Common paradigms are supervised learning, "
            "unsupervised learning, and reinforcement learning."
        ),
    },
    {
        "id":    "C2",
        "title": "Neural networks and deep learning",
        "text": (
            "A neural network is a computational model loosely inspired by the "
            "human brain, composed of layers of interconnected nodes (neurons). "
            "Deep learning refers to neural networks with many hidden layers. "
            "These models excel at image recognition, speech, and language tasks."
        ),
    },
    {
        "id":    "C3",
        "title": "Transformers and attention",
        "text": (
            "The Transformer architecture uses self-attention to weigh the "
            "relevance of each token to every other token in a sequence, enabling "
            "parallel processing. Introduced in 2017, it became the foundation for "
            "models like GPT, BERT, and T5, powering modern NLP breakthroughs."
        ),
    },
    {
        "id":    "C4",
        "title": "Reinforcement learning",
        "text": (
            "Reinforcement learning (RL) trains agents to take actions in an "
            "environment to maximise cumulative reward. A key challenge is the "
            "exploration–exploitation trade-off. RL has achieved superhuman "
            "performance in games like Go and Atari, and underlies RLHF for LLMs."
        ),
    },
    {
        "id":    "C5",
        "title": "AI ethics and fairness",
        "text": (
            "AI ethics addresses concerns about bias, fairness, transparency, and "
            "accountability in AI systems. Models trained on biased data can "
            "perpetuate or amplify discrimination. Fairness metrics, explainability "
            "methods, and diverse training data are key mitigation strategies."
        ),
    },
]


# ---------------------------------------------------------------------------
# 2. RETRIEVER  (Jaccard similarity)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    stop = {
        "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
        "or", "for", "as", "at", "by", "be", "this", "that", "are",
        "with", "its", "was", "can", "more", "from", "into", "such",
        "both", "their", "also", "they", "have", "has", "been", "which",
        "these", "like", "each", "every", "about",
    }
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in stop}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    query_tokens = _tokenize(query)
    scored = []
    for doc in CORPUS:
        doc_tokens = _tokenize(doc["title"] + " " + doc["text"])
        score = jaccard(query_tokens, doc_tokens)
        scored.append({"score": score, "doc": doc})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# 3. DIRECT-ANSWER KNOWLEDGE DICT  (fallback when corpus has no match)
# ---------------------------------------------------------------------------

DIRECT_KNOWLEDGE: dict[str, str] = {
    "python": (
        "Python is a high-level, dynamically typed programming language popular "
        "for data science, scripting, and web development. Its readable syntax "
        "and extensive libraries make it the de-facto language for AI research."
    ),
    "gpu": (
        "GPUs (Graphics Processing Units) accelerate AI training by performing "
        "thousands of matrix multiplications in parallel. NVIDIA's CUDA platform "
        "is the dominant framework for GPU-accelerated deep learning."
    ),
    "cloud": (
        "Cloud platforms (AWS, GCP, Azure) provide on-demand compute for AI "
        "workloads. Managed services like SageMaker, Vertex AI, and Azure ML "
        "abstract infrastructure so teams can focus on modelling."
    ),
    "default": (
        "That is a great question! While it falls outside my local knowledge base, "
        "a real system would query a broader knowledge base or fall back to the "
        "model's parametric knowledge for a general answer."
    ),
}


def direct_answer(query: str) -> str:
    lower = query.lower()
    for keyword, answer in DIRECT_KNOWLEDGE.items():
        if keyword != "default" and keyword in lower:
            return answer
    return DIRECT_KNOWLEDGE["default"]


# ---------------------------------------------------------------------------
# 4. SIMULATED RAG RESPONSES  (keyed to likely queries)
# ---------------------------------------------------------------------------

RAG_RESPONSES: list[dict] = [
    {
        "keywords": ["machine learning", "learn", "supervised", "unsupervised"],
        "answer": (
            "Based on the retrieved context: Machine learning is an AI subfield "
            "where systems improve from data rather than explicit programming. "
            "The three main paradigms are supervised learning (labelled data), "
            "unsupervised learning (unlabelled data), and reinforcement learning "
            "(reward signals from an environment)."
        ),
    },
    {
        "keywords": ["transformer", "attention", "bert", "gpt", "nlp"],
        "answer": (
            "Based on the retrieved context: The Transformer architecture, "
            "introduced in 2017, uses self-attention to process all tokens in a "
            "sequence simultaneously. This enables models like GPT and BERT to "
            "capture long-range dependencies far more efficiently than earlier "
            "recurrent architectures."
        ),
    },
    {
        "keywords": ["reinforcement", "reward", "agent", "rlhf", "game"],
        "answer": (
            "Based on the retrieved context: Reinforcement learning trains agents "
            "to maximise cumulative reward by interacting with an environment. "
            "The key challenge is balancing exploration of new strategies versus "
            "exploitation of known ones. RLHF (RL from Human Feedback) adapts "
            "this idea to fine-tune LLMs."
        ),
    },
    {
        "keywords": ["ethics", "bias", "fair", "transparency", "accountability"],
        "answer": (
            "Based on the retrieved context: AI ethics focuses on preventing bias, "
            "ensuring fairness, and maintaining transparency in AI systems. "
            "Models trained on skewed data can amplify real-world discrimination. "
            "Mitigation approaches include fairness metrics, explainability tools, "
            "and diverse, representative training data."
        ),
    },
    {
        "keywords": ["neural", "deep learning", "layer", "neuron", "image"],
        "answer": (
            "Based on the retrieved context: Neural networks are computational "
            "models with layers of interconnected nodes that transform inputs into "
            "outputs. Deep learning stacks many such layers to learn hierarchical "
            "features automatically. These models are state-of-the-art for vision, "
            "audio, and language tasks."
        ),
    },
]


def rag_answer(query: str, retrieved_docs: list[dict]) -> str:
    lower = query.lower()
    for template in RAG_RESPONSES:
        if any(kw in lower for kw in template["keywords"]):
            return template["answer"]
    # Generic RAG response using first retrieved doc title
    if retrieved_docs:
        title = retrieved_docs[0]["doc"]["title"]
        return (
            f"Based on the retrieved context about '{title}': "
            "The documents contain relevant information. A real LLM would "
            "synthesise a precise answer grounded in the retrieved text."
        )
    return "No relevant context was found for this query."


# ---------------------------------------------------------------------------
# 5. THE AGENT
# ---------------------------------------------------------------------------

@dataclass
class AgentTurn:
    query:          str
    decision:       str           # "RAG" or "DIRECT"
    best_score:     float
    retrieved:      list[dict]    # populated on RAG path, empty on DIRECT
    final_answer:   str
    reasoning:      str


def agent_respond(query: str) -> AgentTurn:
    """Core agent logic: decide path, retrieve if needed, generate answer."""

    # Step 1: Probe the corpus to make a routing decision
    candidates = retrieve(query, top_k=TOP_K)
    best_score  = candidates[0]["score"] if candidates else 0.0

    # Step 2: Route
    if best_score >= ROUTE_THRESHOLD:
        decision  = "RAG"
        reasoning = (
            f"Best Jaccard score {best_score:.4f} >= threshold {ROUTE_THRESHOLD} "
            f"→ corpus has relevant content. Retrieving top-{TOP_K} documents."
        )
        retrieved   = candidates
        final_answer = rag_answer(query, retrieved)
    else:
        decision  = "DIRECT"
        reasoning = (
            f"Best Jaccard score {best_score:.4f} < threshold {ROUTE_THRESHOLD} "
            "→ corpus has little overlap with this query. "
            "Answering from built-in knowledge."
        )
        retrieved   = []
        final_answer = direct_answer(query)

    return AgentTurn(
        query        = query,
        decision     = decision,
        best_score   = best_score,
        retrieved    = retrieved,
        final_answer = final_answer,
        reasoning    = reasoning,
    )


# ---------------------------------------------------------------------------
# 6. DEMO TURNS
# ---------------------------------------------------------------------------

DEMO_TURNS: list[str] = [
    "How does reinforcement learning work?",          # RAG path — strong corpus match
    "What is the Transformer architecture?",          # RAG path — strong corpus match
    "What programming language is best for AI?",     # DIRECT path — not in corpus
    "Tell me about AI bias and fairness.",             # RAG path — medium match
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 70
THIN    = "-" * 70


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=66, initial_indent=indent,
                         subsequent_indent=indent)


def print_turn(turn_num: int, turn: AgentTurn) -> None:
    print()
    print(DIVIDER)
    print(f"  TURN {turn_num}: {turn.query}")
    print(DIVIDER)

    # Decision
    print()
    print("  AGENT DECISION")
    print(THIN)
    print(f"  Path chosen:   {turn.decision}")
    print(f"  Best score:    {turn.best_score:.4f}  (threshold: {ROUTE_THRESHOLD})")
    print(wrap(turn.reasoning, indent="  "))

    # Retrieved docs (RAG only)
    if turn.decision == "RAG":
        print()
        print("  RETRIEVED DOCUMENTS")
        print(THIN)
        for rank, item in enumerate(turn.retrieved, start=1):
            doc = item["doc"]
            print(f"  [{rank}] {doc['id']} — {doc['title']}  (score: {item['score']:.4f})")
            preview = doc["text"][:110].rstrip() + "..."
            print(wrap(preview, indent="      "))
        print()
        print("  CONTEXT INJECTED INTO PROMPT")
        print(THIN)
        for item in turn.retrieved:
            doc = item["doc"]
            print(f"  • {doc['title']}")
    else:
        print()
        print("  KNOWLEDGE PATH")
        print(THIN)
        print("  Corpus had insufficient overlap → using built-in knowledge dict.")

    # Final answer
    print()
    print("  FINAL ANSWER")
    print(THIN)
    print(wrap(turn.final_answer, indent="  "))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 61: MINI END-TO-END RAG + AGENT DEMO")
    print(DIVIDER)
    print()
    print("  System overview")
    print(THIN)
    overview = [
        ("Corpus",     f"{len(CORPUS)} AI topic paragraphs (in-memory)"),
        ("Retriever",  "Jaccard word-overlap similarity"),
        ("Routing",    f"Jaccard >= {ROUTE_THRESHOLD} → RAG path; else → direct path"),
        ("RAG path",   "Retrieve top-2 docs, build context, simulate LLM answer"),
        ("Direct path","Query built-in knowledge dict, return general answer"),
        ("LLM calls",  "Simulated (no API required)"),
    ]
    for name, desc in overview:
        print(f"  {name:<14}  {desc}")

    print()
    print("  CORPUS DOCUMENTS")
    print(THIN)
    for doc in CORPUS:
        print(f"  {doc['id']}  {doc['title']}")

    print()
    print(f"  Running {len(DEMO_TURNS)} demo turns...")

    turns: list[AgentTurn] = []
    for i, query in enumerate(DEMO_TURNS, start=1):
        turn = agent_respond(query)
        turns.append(turn)
        print_turn(i, turn)

    # ---- Summary -----------------------------------------------------------
    print(DIVIDER)
    print("  ROUTING SUMMARY")
    print(DIVIDER)
    print()
    header = f"  {'Turn':<6}  {'Path':<7}  {'Score':<8}  Query"
    print(header)
    print(THIN)
    for i, t in enumerate(turns, start=1):
        score_str = f"{t.best_score:.4f}"
        q_preview = t.query[:44]
        print(f"  {i:<6}  {t.decision:<7}  {score_str:<8}  {q_preview}")
    print()

    rag_count    = sum(1 for t in turns if t.decision == "RAG")
    direct_count = len(turns) - rag_count
    print(f"  RAG path used:    {rag_count}/{len(turns)} turns")
    print(f"  Direct path used: {direct_count}/{len(turns)} turns")

    print()
    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. The routing decision is the agent's core skill — knowing when to",
        "   look something up vs. when to answer from memory.",
        "2. Jaccard similarity works as a routing signal; production systems",
        "   use dense vector embeddings for better recall.",
        "3. The RAG path grounds the answer in retrieved text, reducing",
        "   hallucinations on domain-specific questions.",
        "4. The direct path handles out-of-corpus queries without unnecessary",
        "   retrieval overhead.",
        "5. Threshold tuning is critical: too high → over-routes to direct;",
        "   too low → retrieves irrelevant context.",
        "6. This entire agent ran with zero external dependencies.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
