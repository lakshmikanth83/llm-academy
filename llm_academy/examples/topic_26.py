"""
Topic 26: Embedding Models & Similarity Metrics
=================================================
A from-scratch, offline demo of text embeddings and cosine similarity.
Builds a crude hashing-based bag-of-words vectorizer (no ML, no numpy) to
stand in for real neural embedding models, then uses it to power a mini
nearest-neighbor search over a small corpus.
"""

from __future__ import annotations

import math
import re
import textwrap

DIVIDER = "=" * 70
THIN = "-" * 70
VECTOR_DIM = 128
WRAP_W = 66


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=WRAP_W, initial_indent=indent,
                         subsequent_indent=indent)


# ---------------------------------------------------------------------------
# 1. TOY EMBEDDING FUNCTION — hashing-trick bag-of-words vectorizer
# ---------------------------------------------------------------------------
#
# Real embedding models (word2vec, BERT, OpenAI text-embedding-3, Voyage,
# ...) are neural networks trained on billions of words to learn dense
# vectors where MEANING determines closeness — "dog" and "puppy" end up
# near each other even though the words are spelled completely differently.
#
# We have no neural network here, and no dataset to train one on. Instead
# we build a legitimate, if crude, classical technique: a HASHING-TRICK
# bag-of-words vectorizer. Every word is deterministically hashed into one
# of a fixed number of "buckets" (dimensions), and we count how often each
# bucket gets hit. Sentences that share vocabulary land on the same
# buckets and therefore point in similar directions in vector space.
#
# This is NOT semantic understanding — it is literal word overlap in
# disguise. It cannot tell that "dog" and "canine" mean the same thing.
# That limitation is demonstrated explicitly later in this file.

_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "as", "at", "by", "be", "this", "that", "are", "with", "its",
    "was", "his", "her", "him", "he", "she", "them", "their", "my",
}


def _tokenize(text: str) -> list[str]:
    """
    Lower-case the text, split into alphabetic word tokens, and drop very
    common stop words. Stop-word filtering matters here: without it,
    high-frequency words like "the" and "and" dominate the bucket counts
    and drown out the topical words that actually distinguish sentences.
    """
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOP_WORDS]


def _hash_token(token: str) -> int:
    """
    A small, deterministic string hash (a simple rolling polynomial hash,
    the same idea Java's String.hashCode() uses). We do NOT use Python's
    built-in hash() because CPython randomizes string hashes per process
    for security — that would make this demo's output different every
    run. Determinism matters here so the similarity scores are reproducible.
    """
    h = 0
    for ch in token:
        h = (h * 31 + ord(ch)) % 1_000_003
    return h


def embed(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """
    The toy 'embedding model'. Converts text into a fixed-length vector by:
      1. Lower-casing and tokenizing into words.
      2. Hashing each word into one of `dim` buckets.
      3. Incrementing that bucket's count for every occurrence.

    The result is a term-frequency vector in a hashed, fixed-size space —
    a simplified stand-in for a real dense embedding. Two texts that share
    words will produce vectors that point in similar directions.
    """
    vec = [0.0] * dim
    for token in _tokenize(text):
        bucket = _hash_token(token) % dim
        vec[bucket] += 1.0
    return vec


# ---------------------------------------------------------------------------
# 2. SIMILARITY METRICS — dot product, magnitude, cosine similarity
# ---------------------------------------------------------------------------

def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """A · B = sum of element-wise products."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


def magnitude(vec: list[float]) -> float:
    """|V| = Euclidean length of the vector = sqrt(sum of squares)."""
    return math.sqrt(sum(v * v for v in vec))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    cosine_similarity(A, B) = (A · B) / (|A| * |B|)

    Measures the cosine of the angle between two vectors — i.e. how
    similar their DIRECTION is, ignoring magnitude/length. Range is
    mathematically [-1, 1]; for non-negative count vectors like ours it
    lands in [0, 1]. 1.0 means identical direction (very similar text
    under this scheme), 0.0 means no shared vocabulary at all.
    """
    mag_a = magnitude(vec_a)
    mag_b = magnitude(vec_b)
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product(vec_a, vec_b) / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# 3. CORPUS — a small set of sentences across three clear topics
# ---------------------------------------------------------------------------

CORPUS = [
    "My dog loves to play fetch in the park every morning.",
    "The cat sat quietly on the windowsill watching birds.",
    "I adopted a playful puppy last week and named him Max.",
    "Kittens are playful and love chasing a ball of string.",
    "To make a great pasta sauce, simmer tomatoes with garlic and basil.",
    "Baking bread requires kneading the dough until smooth and elastic.",
    "The chef seasoned the steak with salt, pepper, and rosemary before grilling.",
    "NASA launched a new rocket to explore the surface of Mars.",
    "Astronauts aboard the space station conducted experiments in zero gravity.",
    "The spacecraft's engines ignited for the long journey to the moon.",
]


def show_corpus_embeddings() -> None:
    print("  Embedding every sentence in the corpus")
    print(THIN)
    print(f"  (vectors are {VECTOR_DIM}-dimensional; showing first 8 values)\n")
    for i, sentence in enumerate(CORPUS, start=1):
        vec = embed(sentence)
        preview = ", ".join(f"{v:.0f}" for v in vec[:8])
        print(f"  [{i:>2}] {sentence}")
        print(f"        embedding[:8] = [{preview}, ...]")
    print()


# ---------------------------------------------------------------------------
# 4. NEAREST-NEIGHBOR SEARCH — rank the corpus against a query
# ---------------------------------------------------------------------------

def nearest_neighbors(query: str, corpus: list[str], top_k: int = 3) -> list[tuple[float, str]]:
    """Embed the query and every corpus sentence, rank by cosine similarity."""
    query_vec = embed(query)
    scored = []
    for sentence in corpus:
        score = cosine_similarity(query_vec, embed(sentence))
        scored.append((score, sentence))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


QUERIES = [
    "My puppy is playful and loves to play in the park.",
    "I want to make a pasta sauce with tomatoes and garlic.",
    "The astronauts launched a rocket into space toward the moon.",
]


def show_nearest_neighbor_search() -> None:
    print("  Nearest-neighbor search: rank the whole corpus against each query")
    print(THIN)
    for q_num, query in enumerate(QUERIES, start=1):
        print(f"\n  Query {q_num}: \"{query}\"")
        results = nearest_neighbors(query, CORPUS, top_k=3)
        print(f"  {'Rank':<6}{'Score':<10}Sentence")
        for rank, (score, sentence) in enumerate(results, start=1):
            trimmed = sentence if len(sentence) <= 52 else sentence[:49] + "..."
            print(f"  {rank:<6}{score:<10.4f}{trimmed}")
    print()


# ---------------------------------------------------------------------------
# 5. ILLUSTRATIVE PAIR COMPARISON — honest strengths and limits
# ---------------------------------------------------------------------------

def show_pair_comparisons() -> None:
    print("  Same-topic pair vs. unrelated pair (sentences that SHARE words)")
    print(THIN)

    same_topic_a = "The dog played fetch happily in the sunny park."
    same_topic_b = "My dog loves to play in the park every single day."
    unrelated = "NASA launched a new rocket to explore the surface of Mars."

    sim_same = cosine_similarity(embed(same_topic_a), embed(same_topic_b))
    sim_diff = cosine_similarity(embed(same_topic_a), embed(unrelated))

    print(f'  A: "{same_topic_a}"')
    print(f'  B: "{same_topic_b}"')
    print(f"  cosine_similarity(A, B) = {sim_same:.4f}   (shared words: dog, play, park)")
    print()
    print(f'  A: "{same_topic_a}"')
    print(f'  C: "{unrelated}"')
    print(f"  cosine_similarity(A, C) = {sim_diff:.4f}   (no shared vocabulary)")
    print()
    print(
        wrap(
            "As expected, the pair about the same topic (sharing the words 'dog', "
            "'play', 'park') scores noticeably higher than the unrelated pair. This "
            "is a legitimate result: it comes purely from vocabulary overlap, not "
            "from any understanding of what a dog or a rocket IS."
        )
    )

    print("\n  LIMITATION — this method cannot recognize synonyms")
    print(THIN)
    dog_sentence = "The dog ran across the yard quickly."
    canine_sentence = "The canine ran across the yard quickly."
    sim_synonym_context = cosine_similarity(embed(dog_sentence), embed(canine_sentence))

    dog_only = "Dogs."
    canine_only = "Canines."
    sim_synonym_bare = cosine_similarity(embed(dog_only), embed(canine_only))

    print(f'  "{dog_sentence}"')
    print(f'  "{canine_sentence}"')
    print(f"  cosine_similarity = {sim_synonym_context:.4f}  (high, but ONLY because")
    print("    'ran', 'across', 'yard', 'quickly' are shared — not because 'dog'")
    print("    and 'canine' are recognized as the same concept)")
    print()
    print(f'  "{dog_only}"   vs.   "{canine_only}"')
    print(f"  cosine_similarity = {sim_synonym_bare:.4f}  (near zero — with no other")
    print("    words to overlap on, the hashing vectorizer sees 'dog' and 'canine'")
    print("    as completely unrelated tokens, even though they mean the same thing)")
    print()
    print(
        wrap(
            "A real trained embedding model would place 'dog' and 'canine' near "
            "each other in vector space because it learned their meaning from "
            "context across a huge corpus. Our hashing-based bag-of-words approach "
            "has no notion of meaning at all — it only ever sees exact tokens."
        )
    )
    print()


# ---------------------------------------------------------------------------
# 6. COMPARISON — toy hashing approach vs. real trained embedding models
# ---------------------------------------------------------------------------

def show_comparison_table() -> None:
    print("  Toy hashing embeddings vs. real trained embedding models")
    print(THIN)

    rows = [
        ("Dimensions", f"{VECTOR_DIM} (arbitrary, tiny)", "384 - 3072 (model-defined)"),
        ("Captures synonyms?", "No — exact tokens only", "Yes — learned from context"),
        ("Training required?", "None — deterministic hash", "Yes — trained on huge corpora"),
        ("Word order matters?", "No (bag-of-words)", "Yes (Transformer attention)"),
        ("Compute cost", "Negligible, pure CPU loops", "Neural network inference"),
        ("Typical use", "Teaching / offline demos", "Semantic search, RAG, clustering"),
        ("Example", "This file's embed() function", "OpenAI text-embedding-3, BERT, word2vec"),
    ]

    col1, col2, col3 = 20, 30, 38
    print(f"  {'Aspect':<{col1}} {'Toy Hashing (this demo)':<{col2}} {'Real Embedding Models':<{col3}}")
    print("  " + "-" * (col1 + col2 + col3 + 2))
    for aspect, toy, real in rows:
        print(f"  {aspect:<{col1}} {toy:<{col2}} {real:<{col3}}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 26: EMBEDDING MODELS & SIMILARITY METRICS")
    print(DIVIDER)
    print()
    print(
        wrap(
            "Embeddings turn text into vectors of numbers so that meaning can be "
            "compared mathematically. Real embedding models (word2vec, BERT, "
            "OpenAI/Voyage embeddings) are neural networks trained on enormous "
            "text corpora. We have neither a network nor training data here, so "
            "this demo builds a crude but genuine classical alternative: a "
            "hashing-trick bag-of-words vectorizer, paired with real cosine "
            "similarity math."
        )
    )
    print()

    print(THIN)
    print("  DEMO 1 — Embedding the corpus")
    print(THIN)
    show_corpus_embeddings()

    print(THIN)
    print("  DEMO 2 — Nearest-neighbor search")
    print(THIN)
    show_nearest_neighbor_search()

    print(THIN)
    print("  DEMO 3 — Pair comparisons: what this toy method can and can't do")
    print(THIN)
    show_pair_comparisons()

    print(THIN)
    print("  DEMO 4 — Toy vs. real embedding models")
    print(THIN)
    show_comparison_table()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. An embedding is just a fixed-length numeric vector representing text;",
        "   similar text should produce vectors pointing in similar directions.",
        "2. Cosine similarity = dot(A, B) / (|A| * |B|) — it measures the angle",
        "   between vectors, ignoring their length.",
        "3. Nearest-neighbor search ranks a corpus by cosine similarity to a",
        "   query vector — this is the core operation behind semantic search.",
        "4. Our hashing-based bag-of-words vectorizer is a real, working",
        "   technique — but it only detects shared VOCABULARY, not meaning.",
        "5. It cannot recognize synonyms ('dog' vs 'canine') because it has no",
        "   training signal — it only counts exact hashed tokens.",
        "6. Real models (word2vec, BERT, OpenAI text-embedding-3) learn meaning",
        "   from massive training data, letting synonyms and paraphrases land",
        "   near each other in vector space — something no hashing trick can do.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
