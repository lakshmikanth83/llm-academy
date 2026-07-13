"""
Topic 27: Vector Databases
===========================
Builds a minimal in-memory vector database from scratch — no chromadb,
no numpy, no faiss — to teach the core concepts: embedding + indexing,
metadata storage, and semantic (meaning-based) search.

This is a stand-in for real vector databases like Chroma, Pinecone, or
Weaviate: same ideas (embed -> store -> rank by similarity -> filter by
metadata), just implemented with plain Python lists and dicts so it runs
completely offline with zero installs.
"""

from __future__ import annotations

import hashlib
import math
import re
import textwrap

DIVIDER = "=" * 70
THIN = "-" * 70


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=66, initial_indent=indent, subsequent_indent=indent)


# ---------------------------------------------------------------------------
# 1. TOY EMBEDDING FUNCTION — bag-of-words hashing vectorizer
# ---------------------------------------------------------------------------
# Real embedding models (OpenAI text-embedding-3, sentence-transformers) are
# neural networks trained on billions of examples to place semantically
# similar text near each other in a dense vector space. We can't train one
# here, so we build the simplest stand-in: tokenize -> normalize a few words
# through a tiny synonym table (a crude proxy for what a trained model learns
# automatically) -> hash each word into one of DIM buckets and count
# occurrences -> L2-normalize. The result is a fixed-size numeric
# "fingerprint" where texts sharing (or synonymous) vocabulary point in
# similar directions — just like a real embedding, only much cruder.

# More buckets = fewer accidental hash collisions between unrelated words
# (the classic tradeoff of the "hashing trick" used by real feature-hashed
# vectorizers too).
DIM = 256

_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "as", "at", "by", "be", "this", "that", "are", "with", "its",
    "was", "can", "from", "into", "some", "your", "you", "how", "what",
}

# A hand-written stand-in for the synonym relationships a real embedding
# model would learn on its own from training data (e.g. "homemade" and
# "home-cooked" would naturally land near each other in a real model).
_SYNONYMS = {
    "meal": "dish", "meals": "dish", "dishes": "dish",
    "homemade": "home", "household": "home",
    "cooking": "cook", "cooked": "cook", "cook": "cook",
    "affordable": "budget", "inexpensive": "budget", "cheap": "budget",
    "temple": "shrine", "temples": "shrine",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    tokens = [w for w in words if len(w) > 2 and w not in _STOPWORDS]
    return [_SYNONYMS.get(w, w) for w in tokens]


def _hash_bucket(token: str, dim: int = DIM) -> int:
    """Deterministic hash -> bucket index (stable across runs, unlike hash())."""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


def embed(text: str, dim: int = DIM) -> tuple[float, ...]:
    """Turn text into a fixed-size numeric vector (a toy embedding)."""
    vec = [0.0] * dim
    for token in _tokenize(text):
        vec[_hash_bucket(token, dim)] += 1.0

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return tuple(vec)
    return tuple(v / norm for v in vec)


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity: dot(a, b) / (||a|| * ||b||). 0.0 for zero vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# 2. MINI VECTOR DATABASE
# ---------------------------------------------------------------------------

class MiniVectorDB:
    """
    A minimal, from-scratch vector database. Stores, per document, its raw
    text, embedding vector, and an arbitrary metadata dict — then supports
    similarity search with optional metadata filtering, the two operations
    every real vector DB provides.

    PERFORMANCE NOTE: `query()` is a brute-force O(n) scan — it compares the
    query vector against EVERY stored vector. Fine for a handful of
    documents, but far too slow at millions of vectors. Production vector
    databases (Chroma, Pinecone, Weaviate, Qdrant, pgvector) instead build an
    approximate nearest neighbor (ANN) index — usually HNSW (Hierarchical
    Navigable Small World) or IVF (Inverted File) — to answer the same query
    in roughly O(log n) time by checking only a small neighborhood of
    candidates. This class favors correctness and readability over speed.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self._records: dict[str, dict] = {}  # id -> {text, embedding, metadata}

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        """Embed `text` and store it alongside its id and metadata (an 'upsert')."""
        self._records[doc_id] = {
            "text": text,
            "embedding": embed(text, self.dim),
            "metadata": dict(metadata or {}),
        }

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _passes_filter(metadata: dict, metadata_filter: dict | None) -> bool:
        """Return True if `metadata` satisfies every key/value in the filter."""
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            if metadata.get(key) != expected:
                return False
        return True

    def query(
        self,
        text: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Embed `text`, score it against every stored vector with cosine
        similarity, optionally restrict candidates by metadata, and return
        the top_k highest-scoring documents.

        Filtering is applied *before* scoring (a "pre-filter"), which is
        cheaper — we simply never compute similarity for documents that
        cannot match. Real systems can also post-filter after ranking; the
        tradeoff is discussed in the demo output below.
        """
        query_vec = embed(text, self.dim)

        candidates = [
            (doc_id, record)
            for doc_id, record in self._records.items()
            if self._passes_filter(record["metadata"], metadata_filter)
        ]

        scored = [
            {
                "id": doc_id,
                "text": record["text"],
                "metadata": record["metadata"],
                "score": cosine_similarity(query_vec, record["embedding"]),
            }
            for doc_id, record in candidates
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 3. NAIVE KEYWORD SEARCH — the "before" picture
# ---------------------------------------------------------------------------

def keyword_search(query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
    """
    Traditional full-text search, naive version: a document is a "hit" only
    if the literal query string appears in it (case-insensitive substring
    match). No notion of meaning, synonyms, or word order — an exact string
    either is or isn't there.
    """
    needle = query.lower()
    hits = [doc for doc in documents if needle in doc["text"].lower()]
    return hits[:top_k]


# ---------------------------------------------------------------------------
# 4. SAMPLE DOCUMENT SET
# ---------------------------------------------------------------------------

def _doc(doc_id, text, category, author, date):
    return {"id": doc_id, "text": text, "metadata": {"category": category, "author": author, "date": date}}


DOCUMENTS = [
    _doc("doc01", "Grandma's classic banana bread recipe with walnuts and cinnamon.", "recipe", "Elena", "2024-01-10"),
    _doc("doc02", "How to bake a moist chocolate cake from scratch, step by step.", "recipe", "Marco", "2024-02-14"),
    _doc("doc03", "A beginner's guide to Python list comprehensions and generators.", "tech", "Sam", "2024-03-01"),
    _doc("doc04", "Understanding REST APIs and how servers handle incoming requests.", "tech", "Priya", "2024-03-15"),
    _doc("doc05", "Top five hidden beaches worth visiting in Thailand this summer.", "travel", "Elena", "2024-04-02"),
    _doc("doc06", "Backpacking through the Swiss Alps: a budget travel guide.", "travel", "Marco", "2024-04-20"),
    _doc("doc07", "A simple homemade pasta dish with garlic, olive oil, and parmesan.", "recipe", "Sam", "2024-05-05"),
    _doc("doc08", "Debugging memory leaks in long-running Python backend services.", "tech", "Priya", "2024-05-18"),
    _doc("doc09", "Exploring ancient temples and street food markets in Kyoto.", "travel", "Elena", "2024-06-01"),
    _doc("doc10", "Setting up an approximate nearest neighbor index for fast similarity search.", "tech", "Sam", "2024-06-10"),
]


# ---------------------------------------------------------------------------
# 5. DEMO: INGESTION
# ---------------------------------------------------------------------------

def demo_ingest() -> MiniVectorDB:
    print(DIVIDER)
    print("  DEMO 1: INGESTING DOCUMENTS INTO THE MINI VECTOR DB")
    print(DIVIDER)
    print()
    print("  Each document is embedded (text -> fixed-size numeric vector)")
    print("  and stored together with its metadata — exactly what add()/upsert()")
    print("  does in Chroma, Pinecone, or Weaviate.")
    print()

    db = MiniVectorDB()
    print(f"  {'ID':<8} {'Category':<10} {'Author':<8} Text")
    print(f"  {'-'*6:<8} {'-'*8:<10} {'-'*6:<8} {'-'*40}")
    for doc in DOCUMENTS:
        db.add(doc["id"], doc["text"], doc["metadata"])
        meta = doc["metadata"]
        preview = doc["text"][:40]
        print(f"  {doc['id']:<8} {meta['category']:<10} {meta['author']:<8} {preview}")

    print(f"\n  Stored {len(db)} documents, each as a {DIM}-dimensional vector.")
    print()
    return db


# ---------------------------------------------------------------------------
# 6. DEMO: KEYWORD SEARCH MISSES, SEMANTIC SEARCH FINDS
# ---------------------------------------------------------------------------

def demo_keyword_vs_semantic(db: MiniVectorDB) -> None:
    print(DIVIDER)
    print("  DEMO 2: KEYWORD SEARCH vs. SEMANTIC (VECTOR) SEARCH")
    print(DIVIDER)

    query = "cooking a meal at home tonight"
    print(f'\n  Query: "{query}"\n')

    # --- naive keyword search -------------------------------------------
    print("  STEP A — naive keyword search (literal substring match)")
    print(THIN)
    hits = keyword_search(query, DOCUMENTS)
    if not hits:
        print("  No documents contain the literal phrase above.")
        print("  MISS — doc07 ('A simple homemade pasta dish with garlic...')")
        print("  is highly relevant, but the exact words 'cooking a meal at")
        print("  home tonight' never appear in it, so keyword search cannot")
        print("  find it.")
    else:
        for doc in hits:
            print(f"  Hit: {doc['id']} — {doc['text']}")
    print()

    # --- semantic search ---------------------------------------------------
    print("  STEP B — MiniVectorDB semantic search (cosine similarity)")
    print(THIN)
    results = db.query(query, top_k=3)
    for rank, r in enumerate(results, start=1):
        print(f"  Rank {rank}  score={r['score']:.4f}  {r['id']}  ({r['metadata']['category']})")
        print(wrap(r["text"], indent="      "))
    print()
    print(
        wrap(
            "FOUND — even though the query never appears verbatim in doc07, "
            "'meal' and 'homemade' are normalized to the same buckets as "
            "'dish' and 'home' through the synonym table baked into embed(), "
            "so their vectors point in a similar direction and cosine "
            "similarity correctly ranks doc07 at the top while every "
            "unrelated document scores 0.0. A real embedding model learns "
            "thousands of these relationships automatically from training "
            "data instead of a hand-written table."
        )
    )
    print()


# ---------------------------------------------------------------------------
# 7. DEMO: METADATA FILTERING
# ---------------------------------------------------------------------------

def demo_metadata_filter(db: MiniVectorDB) -> None:
    print(DIVIDER)
    print("  DEMO 3: METADATA FILTERING")
    print(DIVIDER)

    query = "guide to writing better software"
    print(f'\n  Query: "{query}"\n')

    print("  Unfiltered top 3 (all categories eligible):")
    print(THIN)
    for r in db.query(query, top_k=3):
        print(f"  {r['id']:<8} score={r['score']:.4f}  category={r['metadata']['category']:<8} {r['text'][:45]}")

    metadata_filter = {"category": "tech"}
    print(f"\n  Filtered top 3 with metadata_filter={metadata_filter}:")
    print(THIN)
    filtered = db.query(query, top_k=3, metadata_filter=metadata_filter)
    for r in filtered:
        print(f"  {r['id']:<8} score={r['score']:.4f}  category={r['metadata']['category']:<8} {r['text'][:45]}")

    all_tech = all(r["metadata"]["category"] == "tech" for r in filtered)
    print(f"\n  All results are category='tech': {all_tech}")
    print(
        wrap(
            "Metadata filters (category, author, date range, ...) are applied "
            "alongside the vector similarity score. This is how a real system "
            "answers 'find documents about X, but only from the tech category, "
            "written after March 2024' — vector similarity alone cannot express "
            "that kind of exact structured constraint."
        )
    )
    print()


# ---------------------------------------------------------------------------
# 8. COMPARISON: KEYWORD SEARCH vs. VECTOR DATABASE
# ---------------------------------------------------------------------------

def demo_comparison() -> None:
    print(DIVIDER)
    print("  DEMO 4: KEYWORD / FULL-TEXT SEARCH vs. VECTOR DATABASE SEARCH")
    print(DIVIDER)
    print()

    rows = [
        ("Match type", "Exact / substring tokens", "Semantic similarity (meaning)"),
        ("Handles synonyms", "No — 'car' misses 'automobile'", "Yes — similar vectors, similar meaning"),
        ("Needs an embedding model", "No", "Yes — quality depends on the model"),
        ("Result ranking", "Term frequency (e.g. BM25)", "Cosine / dot-product similarity"),
        ("Typical index", "Inverted index", "HNSW, IVF (approximate nearest neighbor)"),
        ("Typical use case", "Log search, exact lookups", "RAG, recommendations, semantic search"),
    ]
    col1, col2, col3 = 24, 32, 38
    print(f"  {'Aspect':<{col1}} {'Keyword Search':<{col2}} {'Vector DB Search':<{col3}}")
    print("  " + "-" * (col1 + col2 + col3))
    for aspect, kw, vec in rows:
        print(f"  {aspect:<{col1}} {kw:<{col2}} {vec:<{col3}}")

    print()
    print(
        wrap(
            "In production, hybrid search (keyword + vector, combined via a "
            "re-ranker) often beats either approach alone — exact matches on "
            "IDs, codes, or names, plus semantic recall for everything else."
        )
    )
    print()
    print("  Production vector databases (this file simulates their core idea):")
    print(THIN)
    dbs = [
        ("Chroma", "Local / embedded, Python-native, great for prototyping"),
        ("Pinecone", "Fully managed cloud service, scales with zero ops"),
        ("Weaviate", "Self-hosted or cloud, built-in hybrid keyword+vector search"),
        ("pgvector", "Adds vector search to an existing PostgreSQL database"),
    ]
    for name, desc in dbs:
        print(f"  {name:<10} {desc}")
    print()
    print(
        wrap(
            "Reach for a real vector database once you have more documents than "
            "fit comfortably in a brute-force scan, need persistence across "
            "restarts, need concurrent read/write access, or need production "
            "features like replication and access control. For a few thousand "
            "documents, a brute-force scan like MiniVectorDB is often plenty."
        )
    )
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 27: VECTOR DATABASES — MINI IN-MEMORY IMPLEMENTATION")
    print(DIVIDER)
    print()
    print(
        wrap(
            "A vector database stores embeddings (numeric fingerprints of "
            "meaning) alongside metadata, and answers queries by finding the "
            "nearest vectors rather than exact keyword matches. This demo "
            "builds one from scratch, using only the Python standard library, "
            "to make every moving part visible."
        )
    )
    print()

    db = demo_ingest()
    demo_keyword_vs_semantic(db)
    demo_metadata_filter(db)
    demo_comparison()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. A vector DB record is (id, embedding vector, metadata, [text]) —",
        "   the same shape whether it's this toy or Chroma/Pinecone/Weaviate.",
        "2. Semantic search ranks by vector similarity (cosine here), so it can",
        "   find relevant documents that share no exact keywords with the query.",
        "3. Metadata filtering narrows the candidate set by exact attributes —",
        "   vector similarity and structured filtering solve different problems.",
        "4. This MiniVectorDB scans every vector (O(n)) for correctness; real",
        "   systems use ANN indexes like HNSW or IVF for O(log n) queries at scale.",
        "5. Embedding quality is everything — this hashing vectorizer is a toy;",
        "   production systems use trained models (OpenAI, sentence-transformers).",
        "6. Keyword and vector search are complementary, not competitors — many",
        "   production systems combine both as hybrid search.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
