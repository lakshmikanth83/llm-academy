"""
Topic 28: Chunking Strategies
=============================
Side-by-side demo of three chunking strategies for RAG pipelines:
fixed-size, recursive, and semantic chunking.

Splits the same sample document three different ways and objectively
measures how cleanly each strategy respects sentence and paragraph
boundaries — no external dependencies, pure Python stdlib.
"""

from __future__ import annotations

import re
import textwrap

DIVIDER = "=" * 70
THIN = "-" * 70
WRAP_W = 66


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=WRAP_W, initial_indent=indent,
                          subsequent_indent=indent)


def print_header(title: str) -> None:
    print(DIVIDER)
    print(f"  {title}")
    print(DIVIDER)


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# SAMPLE DOCUMENT — a short article about RAG, covering three sub-topics
# (retrieval, generation, evaluation) across five paragraphs.
# ---------------------------------------------------------------------------

DOCUMENT = (
    "Retrieval-Augmented Generation, or RAG, has become one of the most "
    "practical patterns for building AI systems that answer questions using "
    "fresh or private information. Instead of relying only on what a "
    "language model memorized during training, RAG lets the model look up "
    "relevant facts at query time and use them as grounding context before "
    "it writes an answer. This combination of search and generation is now "
    "the backbone of most enterprise chatbots, coding assistants, and "
    "customer support tools, because it lets teams update the knowledge a "
    "system uses without ever retraining the underlying model."
    "\n\n"
    "The first stage of a RAG pipeline is retrieval. A large document "
    "collection is split into chunks, each chunk is converted into a numeric "
    "vector using an embedding model, and the vectors are stored in a vector "
    "database. When a user asks a question, the question itself is embedded "
    "and compared against every stored chunk using a similarity metric such "
    "as cosine similarity. The top few chunks, typically three to eight, are "
    "pulled back and treated as the most relevant evidence for answering the "
    "question. If retrieval pulls back the wrong chunks, everything "
    "downstream suffers, no matter how capable the language model is."
    "\n\n"
    "Once the relevant chunks are retrieved, the generation stage begins. "
    "The retrieved text is inserted into a prompt template alongside the "
    "original question, and the language model is instructed to answer "
    "using only the supplied context. This grounding step is what separates "
    "RAG from a model simply guessing from memory, since the model can "
    "quote, summarize, or reason over the retrieved passages instead of "
    "hallucinating facts. Good prompt design matters here, because the model "
    "needs clear instructions to say it does not know rather than inventing "
    "a plausible-sounding response when the context lacks the answer."
    "\n\n"
    "Evaluating a RAG system requires looking at both stages separately. "
    "Retrieval quality is usually measured with metrics like recall at k, "
    "checking whether the correct source chunk appears anywhere in the "
    "top-k results. Generation quality is measured with metrics like "
    "faithfulness, which asks whether every claim in the answer is actually "
    "supported by the retrieved context, and answer relevance, which asks "
    "whether the answer actually addresses the question. A system can have "
    "perfect retrieval and still produce a bad answer if generation ignores "
    "the context, and it can have excellent generation that is worthless if "
    "retrieval never finds the right documents in the first place."
    "\n\n"
    "In production, teams tune RAG systems by adjusting chunk size, overlap, "
    "and the number of retrieved chunks, then measuring the effect on both "
    "recall and faithfulness. Smaller chunks tend to improve retrieval "
    "precision but may lack context, while larger chunks preserve context "
    "but dilute the embedding with multiple topics. Many teams also add a "
    "re-ranking step, using a more expensive but more accurate model to "
    "reorder the top retrieved candidates before they reach generation. None "
    "of these choices exist in isolation. How the original document was "
    "chunked in the first place shapes every downstream decision, which is "
    "exactly why chunking strategy deserves careful attention."
)


# ---------------------------------------------------------------------------
# STRATEGY 1: FIXED-SIZE CHUNKING
# ---------------------------------------------------------------------------

def fixed_size_chunk(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """
    Naive fixed-character-count splitting with overlap.

    Ignores every sentence and paragraph boundary — it simply counts
    characters and cuts. This is the simplest possible strategy and the
    fastest to implement, but it will visibly slice words and sentences
    in half.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# STRATEGY 2: RECURSIVE CHUNKING
# ---------------------------------------------------------------------------

# Separator hierarchy, coarsest first: (regex pattern, string used to
# rejoin pieces that end up sharing one chunk). We fall back to the next
# separator only when the current one fails to produce small-enough
# pieces. The sentence pattern uses a look-behind so the sentence-ending
# punctuation (".", "!", "?") stays attached to the piece it belongs to,
# instead of being stripped the way a plain str.split(". ") would.
_RECURSIVE_SEPARATORS = [
    (r"\n\n", "\n\n"),          # 1. paragraph breaks
    (r"(?<=[.!?])\s+", " "),    # 2. sentence breaks
    (r"\s+", " "),              # 3. word breaks
]


def _split_on_separator(text: str, max_chars: int, separators: list[tuple[str, str]]) -> list[str]:
    """
    Recursive-descent helper: split `text` on the first usable separator,
    then re-pack the resulting pieces into chunks no larger than
    max_chars. Any piece that is still too big after splitting is handed
    back into this same function with the remaining (finer) separators.
    A hard character cut is the last resort once separators run out.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    if not separators:
        # Last resort: no separator left to try — hard character break.
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    (pattern, join_str), rest = separators[0], separators[1:]
    pieces = [p for p in re.split(pattern, text) if p]

    if len(pieces) == 1:
        # This separator does not occur in the text at all — try the
        # next, finer-grained separator instead.
        return _split_on_separator(text, max_chars, rest)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{join_str}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue

        # Adding `piece` would overflow — flush what we have first.
        if current:
            chunks.append(current)
            current = ""

        if len(piece) > max_chars:
            # Even a single piece is too big; recurse with finer separators.
            chunks.extend(_split_on_separator(piece, max_chars, rest))
        else:
            current = piece

    if current:
        chunks.append(current)
    return chunks


def recursive_chunk(text: str, max_chars: int = 200) -> list[str]:
    """
    Real recursive-descent splitter, conceptually mirroring LangChain's
    RecursiveCharacterTextSplitter:

      1. Try splitting on paragraph breaks ("\\n\\n").
      2. If a resulting piece is still too big, try sentence breaks (". ").
      3. If still too big, try word breaks (" ").
      4. If a single "word" is still too big, hard-cut by character count.

    Because it prefers the coarsest structural boundary that fits, most
    chunks end up starting and ending on clean paragraph or sentence
    edges instead of mid-word.
    """
    return _split_on_separator(text.strip(), max_chars, list(_RECURSIVE_SEPARATORS))


# ---------------------------------------------------------------------------
# STRATEGY 3: SEMANTIC CHUNKING
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "as", "at", "by", "be", "this", "that", "are", "with", "its",
    "was", "can", "more", "from", "into", "such", "both", "their", "also",
    "they", "have", "has", "been", "will", "were", "not", "but", "than",
    "then", "which", "when", "what", "these", "those", "does",
}


def _keywords(text: str) -> set[str]:
    """Lower-case content words, used as a cheap stand-in for an embedding."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """|A ∩ B| / |A ∪ B|. Returns 0.0 when both sets are empty."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def semantic_chunk(text: str, similarity_threshold: float = 0.10) -> list[str]:
    """
    Group text by paragraph, then merge adjacent paragraphs into a single
    chunk while a real similarity signal says they discuss the same
    sub-topic. Here that signal is Jaccard overlap between each
    paragraph's keyword set — a bag-of-words proxy for the cosine
    similarity a real system would compute over sentence/paragraph
    embeddings. Paragraphs whose keyword overlap falls below the
    threshold mark a topic shift and start a new chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_chunk = paragraphs[0]
    current_keywords = _keywords(paragraphs[0])

    for para in paragraphs[1:]:
        para_keywords = _keywords(para)
        sim = jaccard_similarity(current_keywords, para_keywords)
        if sim >= similarity_threshold:
            # Similar enough to the running chunk — treat as the same topic.
            current_chunk = f"{current_chunk}\n\n{para}"
            current_keywords = current_keywords | para_keywords
        else:
            # Topic shift detected — close the current chunk, start a new one.
            chunks.append(current_chunk)
            current_chunk = para
            current_keywords = para_keywords

    chunks.append(current_chunk)
    return chunks


# ---------------------------------------------------------------------------
# BOUNDARY-QUALITY MEASUREMENT
# ---------------------------------------------------------------------------

def is_clean_boundary(chunk: str) -> bool:
    """
    A chunk has a "clean" boundary when it starts with an upper-case
    letter (a fresh sentence) and ends with sentence-final punctuation
    (a complete sentence). Anything else means a word or sentence was
    cut in half at that edge.
    """
    stripped = chunk.strip()
    if not stripped:
        return False
    starts_clean = stripped[0].isupper()
    ends_clean = stripped[-1] in ".!?"
    return starts_clean and ends_clean


def boundary_marker(chunk: str) -> str:
    return "[CLEAN BOUNDARY]" if is_clean_boundary(chunk) else "[CUT MID-SENTENCE]"


# ---------------------------------------------------------------------------
# DEMO: RUN + PRINT EACH STRATEGY
# ---------------------------------------------------------------------------

def preview(chunk: str, width: int = 90) -> str:
    flat = " ".join(chunk.split())
    return flat if len(flat) <= width else flat[:width].rstrip() + "..."


def run_strategy(name: str, chunks: list[str]) -> dict:
    print_section(f"{name} — {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks, start=1):
        marker = boundary_marker(chunk)
        print(f"  Chunk {i:2d} | {len(chunk):4d} chars | {marker}")
        print(wrap(f'"{preview(chunk)}"', indent="      "))

    clean = sum(1 for c in chunks if is_clean_boundary(c))
    total_len = sum(len(c) for c in chunks)
    avg_len = total_len / len(chunks) if chunks else 0.0
    return {
        "name": name,
        "count": len(chunks),
        "avg_len": avg_len,
        "clean": clean,
        "mid_cut": len(chunks) - clean,
    }


def print_summary_table(stats: list[dict]) -> None:
    print_section("SUMMARY — chunk count, average size, boundary quality")
    col = (22, 10, 14, 14, 14)
    header = (
        f"  {'Strategy':<{col[0]}} {'# Chunks':<{col[1]}} {'Avg Size':<{col[2]}}"
        f" {'Clean Bound.':<{col[3]}} {'Mid-Sentence':<{col[4]}}"
    )
    print(header)
    print("  " + "-" * (sum(col) + len(col)))
    for s in stats:
        print(
            f"  {s['name']:<{col[0]}} {s['count']:<{col[1]}} "
            f"{s['avg_len']:<{col[2]}.1f} {s['clean']:<{col[3]}} {s['mid_cut']:<{col[4]}}"
        )


def print_comparison_table() -> None:
    print_section("COMPARISON — fixed-size vs recursive vs semantic")
    rows = [
        ("Implementation",   "Trivial: slice by chars",     "Moderate: separator hierarchy",  "Complex: needs a similarity signal"),
        ("Respects meaning", "No — cuts anywhere",          "Mostly — prefers sentence edges", "Yes — merges paragraphs by topic"),
        ("Chunk size",       "Perfectly uniform",            "Bounded, slightly variable",     "Fully variable"),
        ("Best used when",   "Quick prototypes, plain text", "General-purpose RAG (default)",  "Long docs with clear topic shifts"),
    ]
    col1, col2, col3, col4 = 18, 30, 32, 36
    print(f"  {'Aspect':<{col1}} {'Fixed-Size':<{col2}} {'Recursive':<{col3}} {'Semantic':<{col4}}")
    print("  " + "-" * (col1 + col2 + col3 + col4))
    for aspect, fixed, recursive, semantic in rows:
        print(f"  {aspect:<{col1}} {fixed:<{col2}} {recursive:<{col3}} {semantic:<{col4}}")
    print()

    print(
        wrap(
            "Choosing chunk size and overlap for RAG in practice: start around "
            "256-512 tokens (roughly 1000-2000 characters) for dense retrieval "
            "where precision matters, and 512-1024 tokens when the generator "
            "needs more surrounding context. Set overlap to about 10-20% of "
            "chunk size so a sentence that spans a boundary is still fully "
            "present in at least one chunk. Measure with retrieval recall@k on "
            "a labeled query set, then adjust — there is no universally correct "
            "size, only the size that measurably improves your own retrieval "
            "quality.",
            indent="  ",
        )
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print_header("TOPIC 28: CHUNKING STRATEGIES")
    print()
    print(
        "  Before a document can be embedded and searched, it must be cut\n"
        "  into chunks. HOW you cut it changes retrieval quality dramatically.\n"
        "  This demo splits the SAME document three ways and measures the\n"
        "  boundary quality of each result — not just claims it, but counts it."
    )

    print()
    print_section("SOURCE DOCUMENT")
    for para in DOCUMENT.split("\n\n"):
        print(wrap(para))
        print()
    print(f"  Total length: {len(DOCUMENT)} characters, "
          f"{len(DOCUMENT.split())} words, "
          f"{len(DOCUMENT.split(chr(10)*2))} paragraphs.")

    print()
    print_header("STRATEGY 1 — FIXED-SIZE CHUNKING (chunk_size=200, overlap=20)")
    fixed_chunks = fixed_size_chunk(DOCUMENT, chunk_size=200, overlap=20)
    fixed_stats = run_strategy("Fixed-size", fixed_chunks)

    print()
    print_header("STRATEGY 2 — RECURSIVE CHUNKING (max_chars=200)")
    recursive_chunks = recursive_chunk(DOCUMENT, max_chars=200)
    recursive_stats = run_strategy("Recursive", recursive_chunks)

    print()
    print_header("STRATEGY 3 — SEMANTIC CHUNKING (paragraph + keyword-overlap merge)")
    semantic_chunks = semantic_chunk(DOCUMENT, similarity_threshold=0.10)
    semantic_stats = run_strategy("Semantic", semantic_chunks)

    print()
    print(DIVIDER)
    print_summary_table([fixed_stats, recursive_stats, semantic_stats])

    print()
    print(DIVIDER)
    print_comparison_table()

    print()
    print_header("KEY TAKEAWAYS")
    takeaways = [
        "1. Fixed-size chunking is fast to write but blind — it cuts words and",
        "   sentences in half at almost every boundary, as measured above.",
        "2. Recursive chunking respects structure by trying paragraph, then",
        "   sentence, then word breaks before ever hard-cutting a character.",
        "3. Semantic chunking groups by meaning, not size — real systems use",
        "   sentence embeddings; this demo used Jaccard keyword overlap as a",
        "   free, offline stand-in for that similarity signal.",
        "4. Boundary quality is measurable, not just aesthetic: count how many",
        "   chunks start/end on a full sentence versus mid-word.",
        "5. In production RAG, start with recursive chunking at 256-512 tokens",
        "   and 10-20% overlap, then tune using retrieval recall on real queries.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
