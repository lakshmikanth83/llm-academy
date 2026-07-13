"""
Topic 29: Retrieval-Augmented Generation (RAG) — Offline Pipeline
==================================================================
A complete RAG pipeline using only Python stdlib.

Pipeline stages:
  1. Knowledge base  — 10 facts stored as plain strings
  2. Retrieval       — Jaccard similarity (word overlap) as a proxy for
                       semantic similarity; returns top-2 documents
  3. Prompt building — retrieved context is injected into the prompt
  4. Generation      — simulated LLM response (template-based)
  5. Demo            — 3 different queries walk through every stage
"""

from __future__ import annotations

import re
import textwrap

# ---------------------------------------------------------------------------
# 1. KNOWLEDGE BASE
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE: list[dict] = [
    {
        "id": "doc_01",
        "title": "What is a Large Language Model",
        "text": (
            "A Large Language Model (LLM) is a neural network trained on massive "
            "text corpora to predict and generate human-like text. Modern LLMs use "
            "the Transformer architecture and can contain hundreds of billions of "
            "parameters."
        ),
    },
    {
        "id": "doc_02",
        "title": "The Transformer Architecture",
        "text": (
            "The Transformer architecture, introduced in the 2017 paper 'Attention "
            "Is All You Need', relies on self-attention mechanisms to process "
            "sequences in parallel. It replaced recurrent networks for most NLP "
            "tasks and became the foundation for GPT, BERT, and similar models."
        ),
    },
    {
        "id": "doc_03",
        "title": "Tokenization in LLMs",
        "text": (
            "Tokenization is the process of splitting text into sub-word units "
            "called tokens before it enters an LLM. Common algorithms include "
            "Byte-Pair Encoding (BPE) and WordPiece. A typical English word maps "
            "to 1–3 tokens; the average is roughly 0.75 words per token."
        ),
    },
    {
        "id": "doc_04",
        "title": "Prompt Engineering",
        "text": (
            "Prompt engineering is the practice of carefully designing the input "
            "text (prompt) to guide an LLM toward the desired output. Techniques "
            "include zero-shot, few-shot, chain-of-thought, and role prompting. "
            "Well-crafted prompts can dramatically improve response quality without "
            "changing model weights."
        ),
    },
    {
        "id": "doc_05",
        "title": "Retrieval-Augmented Generation (RAG)",
        "text": (
            "RAG combines a retrieval system with a generative model. At query "
            "time, relevant documents are fetched from a knowledge base and "
            "inserted into the prompt as context. This lets the LLM answer "
            "questions about facts it was never trained on and reduces hallucinations."
        ),
    },
    {
        "id": "doc_06",
        "title": "Hallucinations in LLMs",
        "text": (
            "Hallucination refers to an LLM confidently generating text that is "
            "factually incorrect or completely fabricated. It arises because the "
            "model predicts plausible-sounding tokens rather than verifying facts. "
            "Mitigation strategies include RAG, fine-tuning, and self-consistency checks."
        ),
    },
    {
        "id": "doc_07",
        "title": "Fine-Tuning vs. Prompting",
        "text": (
            "Fine-tuning updates a pre-trained model's weights on a smaller, "
            "task-specific dataset. It is more powerful than prompting alone but "
            "requires labelled data, compute, and careful regularisation to avoid "
            "catastrophic forgetting. Prompting is cheaper and faster to iterate on."
        ),
    },
    {
        "id": "doc_08",
        "title": "Embeddings and Vector Databases",
        "text": (
            "Text embeddings are dense numeric vectors that encode semantic meaning. "
            "Similar texts land near each other in vector space. Vector databases "
            "such as Pinecone, Weaviate, and Chroma store these vectors and support "
            "fast approximate nearest-neighbour search, which is the retrieval step "
            "in production RAG systems."
        ),
    },
    {
        "id": "doc_09",
        "title": "Context Window",
        "text": (
            "The context window is the maximum number of tokens an LLM can process "
            "in a single forward pass, covering both the prompt and the generated "
            "response. Early GPT models had a 2 k-token limit; modern models support "
            "128 k or more. RAG must fit retrieved chunks within this budget."
        ),
    },
    {
        "id": "doc_10",
        "title": "AI Safety and Alignment",
        "text": (
            "AI alignment research aims to ensure that AI systems behave in ways "
            "that are safe, beneficial, and consistent with human values. Key "
            "challenges include specifying objectives precisely, avoiding reward "
            "hacking, and ensuring models remain controllable as they become more "
            "capable. Techniques include RLHF and constitutional AI."
        ),
    },
]


# ---------------------------------------------------------------------------
# 2. RETRIEVER — Jaccard similarity (word overlap)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lower-case and split into alphabetic words; remove short stop words."""
    stop = {
        "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
        "or", "for", "as", "at", "by", "be", "this", "that", "are",
        "with", "its", "was", "can", "more", "from", "into", "such",
        "both", "their", "also", "they", "have", "has", "been",
    }
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in stop}


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|.  Returns 0 when both sets are empty."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """Score every document against the query and return the top-k matches."""
    query_tokens = _tokenize(query)
    scored = []
    for doc in KNOWLEDGE_BASE:
        doc_tokens = _tokenize(doc["title"] + " " + doc["text"])
        score = jaccard(query_tokens, doc_tokens)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, "doc": d} for s, d in scored[:top_k]]


# ---------------------------------------------------------------------------
# 3. PROMPT BUILDER
# ---------------------------------------------------------------------------

def build_rag_prompt(query: str, retrieved: list[dict]) -> str:
    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        doc = item["doc"]
        context_blocks.append(
            f"[Document {i} — {doc['title']}]\n{doc['text']}"
        )
    context = "\n\n".join(context_blocks)

    return (
        "You are a helpful AI assistant. Use only the context below to answer "
        "the question. If the answer is not in the context, say so.\n\n"
        "--- CONTEXT ---\n"
        f"{context}\n"
        "--- END CONTEXT ---\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# 4. GENERATOR — simulated responses keyed to each demo query
# ---------------------------------------------------------------------------

SIMULATED_ANSWERS = {
    "what is rag and how does it reduce hallucinations": (
        "RAG (Retrieval-Augmented Generation) combines a retrieval system with a "
        "generative model. At query time it fetches relevant documents and injects "
        "them into the prompt as context, so the model can answer questions about "
        "facts it was never trained on. Because the answer is grounded in retrieved "
        "text rather than learned weights, the model is less likely to fabricate "
        "information — directly reducing hallucinations."
    ),
    "how do transformers use attention": (
        "Transformers rely on self-attention mechanisms to process sequences in "
        "parallel. This mechanism, introduced in the 2017 paper 'Attention Is All "
        "You Need', lets each position in a sequence attend to every other position "
        "simultaneously, capturing long-range dependencies without recurrence. "
        "This parallel processing makes training far more efficient than earlier "
        "RNN-based architectures."
    ),
    "what is the difference between fine tuning and prompting": (
        "Fine-tuning updates a pre-trained model's weights on a smaller, "
        "task-specific dataset, making it more powerful for that task but requiring "
        "labelled data, compute, and care to avoid catastrophic forgetting. "
        "Prompting, by contrast, guides the frozen model through the input text "
        "alone. Prompting is cheaper and faster to iterate on, while fine-tuning "
        "is preferred when consistent, deep task adaptation is needed."
    ),
}


def generate(query: str, prompt: str) -> str:
    """Return a simulated answer.  In production this calls an LLM API."""
    key = re.sub(r"[^a-z0-9 ]", "", query.lower()).strip()
    # Try exact key first, then partial match
    if key in SIMULATED_ANSWERS:
        return SIMULATED_ANSWERS[key]
    for k, v in SIMULATED_ANSWERS.items():
        if any(w in key for w in k.split()):
            return v
    return (
        "[Simulated] Based on the retrieved context, a relevant answer would be "
        "constructed here using only the provided documents."
    )


# ---------------------------------------------------------------------------
# 5. DEMO RUNNER
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "What is RAG and how does it reduce hallucinations?",
    "How do Transformers use attention?",
    "What is the difference between fine-tuning and prompting?",
]

DIVIDER = "=" * 70
THIN    = "-" * 70
WRAP_W  = 66


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=WRAP_W, initial_indent=indent,
                         subsequent_indent=indent)


def run_demo(query: str, query_num: int) -> None:
    print(DIVIDER)
    print(f"  QUERY {query_num}: {query}")
    print(DIVIDER)

    # ---- STEP 1: Retrieve ------------------------------------------------
    print()
    print("  STEP 1 — RETRIEVE  (Jaccard similarity against knowledge base)")
    print(THIN)
    results = retrieve(query, top_k=2)
    for rank, item in enumerate(results, start=1):
        doc   = item["doc"]
        score = item["score"]
        print(f"  Rank {rank}  |  Score: {score:.4f}  |  {doc['id']}: {doc['title']}")
        preview = doc["text"][:120].rstrip() + "..."
        print(wrap(preview))
        print()

    # ---- STEP 2: Build prompt --------------------------------------------
    print("  STEP 2 — BUILD PROMPT  (inject retrieved docs as context)")
    print(THIN)
    prompt = build_rag_prompt(query, results)
    # Show a trimmed view of the prompt
    prompt_lines = prompt.splitlines()
    for line in prompt_lines[:8]:
        print(f"  {line}")
    if len(prompt_lines) > 8:
        omitted = len(prompt_lines) - 8
        print(f"  ... [{omitted} more lines — full context window omitted for brevity]")
    print()

    # ---- STEP 3: Generate ------------------------------------------------
    print("  STEP 3 — GENERATE  (LLM produces answer using context)")
    print(THIN)
    answer = generate(query, prompt)
    print(wrap(answer))
    print()

    # ---- STEP 4: Final answer --------------------------------------------
    print("  FINAL ANSWER")
    print(THIN)
    print(wrap(answer))
    print()


def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 29: RETRIEVAL-AUGMENTED GENERATION (RAG) — OFFLINE DEMO")
    print(DIVIDER)
    print()
    print("  Pipeline overview")
    print(THIN)
    stages = [
        ("Knowledge base",  "10 plain-text documents stored in memory"),
        ("Embedding proxy", "Jaccard word-overlap similarity (no ML required)"),
        ("Retrieval",       "Score all docs, return top-2 matches"),
        ("Prompt building", "Inject retrieved docs as context before the question"),
        ("Generation",      "LLM (simulated here) answers using only the context"),
    ]
    for name, desc in stages:
        print(f"  {name:<20}  {desc}")
    print()
    print(f"  Knowledge base size: {len(KNOWLEDGE_BASE)} documents")
    print()

    for i, query in enumerate(DEMO_QUERIES, start=1):
        run_demo(query, i)

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. RAG decouples knowledge from model weights — update the KB, not the model.",
        "2. Retrieval quality gates answer quality: garbage in, garbage out.",
        "3. Jaccard works as a demo; production systems use dense vector embeddings.",
        "4. The prompt must fit all retrieved chunks inside the context window.",
        "5. Grounding answers in retrieved text measurably reduces hallucinations.",
        "6. This entire pipeline ran with zero external dependencies.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
