"""
Topic 07: Embeddings
=====================
Demo of bag-of-words TF-IDF vectors, cosine similarity matrix,
and nearest-neighbour search — using only Python stdlib.
"""

import math
import re
from collections import Counter


def print_header(title, width=66):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# CORPUS
# ---------------------------------------------------------------------------

SENTENCES = [
    "The cat sat on the warm mat",
    "Dogs and cats are popular pets",
    "Machine learning models learn from data",
    "Deep learning is a subset of machine learning",
    "The dog ran across the garden",
    "Neural networks are used in deep learning models",
]

QUERY = "supervised learning with neural networks"


# ---------------------------------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "can", "could",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "if", "so", "yet", "nor",
}


def tokenize(text):
    """Lowercase, strip punctuation, split, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return [w for w in text.split() if w not in STOPWORDS]


# ---------------------------------------------------------------------------
# VOCABULARY
# ---------------------------------------------------------------------------

def build_vocab(sentences):
    """Return sorted list of unique terms across all sentences."""
    all_words = []
    for s in sentences:
        all_words.extend(tokenize(s))
    vocab = sorted(set(all_words))
    return vocab


# ---------------------------------------------------------------------------
# TF-IDF VECTORS
# ---------------------------------------------------------------------------

def term_frequency(tokens, vocab):
    """TF: count of each vocab term in this document's token list."""
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    return {term: counts.get(term, 0) / total for term in vocab}


def inverse_document_frequency(sentences, vocab):
    """
    IDF: log( N / (1 + number_of_docs_containing_term) )
    +1 smoothing prevents division by zero for unseen terms.
    """
    N = len(sentences)
    idf = {}
    for term in vocab:
        df = sum(1 for s in sentences if term in tokenize(s))
        idf[term] = math.log(N / (1 + df))
    return idf


def tfidf_vector(sentence, vocab, idf):
    """Compute TF-IDF vector as a list aligned to vocab order."""
    tokens = tokenize(sentence)
    tf = term_frequency(tokens, vocab)
    return [tf[term] * idf[term] for term in vocab]


# ---------------------------------------------------------------------------
# COSINE SIMILARITY
# ---------------------------------------------------------------------------

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(v):
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1, v2):
    mag1, mag2 = magnitude(v1), magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)


# ---------------------------------------------------------------------------
# DEMO FUNCTIONS
# ---------------------------------------------------------------------------

def demo_vocabulary(vocab, sentences):
    print_section("STEP 1: Build Vocabulary")
    print(
        "  Every unique (non-stopword) word across all sentences\n"
        "  becomes a dimension in our vector space.\n"
    )
    for s in sentences:
        tokens = tokenize(s)
        print(f"  '{s}'")
        print(f"    tokens: {tokens}")
    print(f"\n  Vocabulary ({len(vocab)} terms):")
    # Print in rows of 8
    for i in range(0, len(vocab), 8):
        chunk = vocab[i:i+8]
        print("    " + "  ".join(f"{w:<12}" for w in chunk))


def demo_tf_idf(sentences, vocab, idf, vectors):
    print_section("STEP 2: TF-IDF Vectors")
    print(
        "  Each sentence becomes a vector of TF-IDF weights.\n"
        "  High weight = term is frequent in this sentence AND rare overall.\n"
        "  Zero weight = term absent from this sentence.\n"
    )

    # Show which terms have high IDF (rare = informative)
    top_idf = sorted(idf.items(), key=lambda x: -x[1])[:8]
    print("  Most informative terms (highest IDF = appear in fewest docs):")
    for term, val in top_idf:
        print(f"    {term:<15} idf={val:.3f}")

    print("\n  Non-zero TF-IDF weights per sentence:")
    for i, (sent, vec) in enumerate(zip(sentences, vectors)):
        nonzero = [(vocab[j], round(v, 3)) for j, v in enumerate(vec) if v > 0]
        print(f"\n  [{i}] '{sent}'")
        if nonzero:
            for term, val in sorted(nonzero, key=lambda x: -x[1]):
                bar = "+" * int(val * 60)
                print(f"      {term:<15} {val:.3f}  {bar}")
        else:
            print("      (all stopwords — zero vector)")


def demo_similarity_matrix(sentences, vectors):
    print_section("STEP 3: Cosine Similarity Matrix")
    print(
        "  Cosine similarity measures the angle between two vectors:\n"
        "    1.0 = identical direction (same meaning)\n"
        "    0.0 = perpendicular (no shared terms)\n"
        "   -1.0 = opposite direction (not possible here; TF-IDF >= 0)\n"
    )

    n = len(sentences)
    labels = [f"S{i}" for i in range(n)]

    # Build matrix
    matrix = [[cosine_similarity(vectors[i], vectors[j]) for j in range(n)]
              for i in range(n)]

    # Print header
    print(f"       {'':5}" + "".join(f"{lb:>7}" for lb in labels))
    print(f"       {'':-<5}" + "-" * (7 * n))
    for i, row in enumerate(matrix):
        row_str = "".join(f"{v:>7.3f}" for v in row)
        sent_short = sentences[i][:30] + ("..." if len(sentences[i]) > 30 else "")
        print(f"  {labels[i]:>4} |{row_str}   {sent_short}")

    # Highlight most similar pair (excluding diagonal)
    best_i, best_j, best_sim = 0, 1, -1
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > best_sim:
                best_sim = matrix[i][j]
                best_i, best_j = i, j

    print(f"\n  Most similar pair: S{best_i} and S{best_j}  (similarity={best_sim:.3f})")
    print(f"    S{best_i}: '{sentences[best_i]}'")
    print(f"    S{best_j}: '{sentences[best_j]}'")


def demo_nearest_neighbour(query, sentences, vocab, idf, vectors):
    print_section("STEP 4: Nearest-Neighbour Search (Query)")
    print(f"  Query: '{query}'\n")

    query_vec = tfidf_vector(query, vocab, idf)
    query_tokens = tokenize(query)
    query_nonzero = [(vocab[j], round(v, 3)) for j, v in enumerate(query_vec) if v > 0]

    print(f"  Query tokens: {query_tokens}")
    print(f"  Query vector (non-zero terms):")
    for term, val in sorted(query_nonzero, key=lambda x: -x[1]):
        print(f"    {term:<15} {val:.3f}")

    print("\n  Similarities to each sentence:")
    scores = []
    for i, (sent, vec) in enumerate(zip(sentences, vectors)):
        sim = cosine_similarity(query_vec, vec)
        scores.append((sim, i, sent))
    scores.sort(reverse=True)

    for rank, (sim, idx, sent) in enumerate(scores, 1):
        bar = "#" * int(sim * 40)
        print(f"  #{rank}  sim={sim:.3f}  [{idx}] '{sent}'")
        print(f"          {'':5} |{bar}")

    best_sim, best_idx, best_sent = scores[0]
    print(f"\n  Best match: '{best_sent}'")
    print(f"  Similarity: {best_sim:.3f}")

    print(
        "\n  Why this works:\n"
        "  The query and the best-matching sentence share key terms.\n"
        "  TF-IDF downweights common words ('the', 'is') and upweights\n"
        "  rare, informative ones ('neural', 'learning', 'networks').\n"
        "  Cosine similarity finds the sentence pointing in the same\n"
        "  direction through this high-dimensional word space."
    )


def demo_intuition(vocab):
    print_section("STEP 5: Embedding Intuition")
    print(
        "  A bag-of-words vector has one dimension per vocabulary word.\n"
        "  For our corpus that is only a few dozen dimensions.\n"
        "  Real LLM embeddings use 768–4096 dimensions, but the idea\n"
        "  is the same: map text to a point in space where similarity\n"
        "  in meaning = closeness in space.\n"
    )
    print(
        "  Limitations of bag-of-words embeddings:\n"
        "    - Word ORDER is ignored ('dog bites man' == 'man bites dog')\n"
        "    - Synonyms are far apart ('happy' vs 'joyful' = 0 similarity)\n"
        "    - Vectors are sparse (mostly zeros)\n"
        "\n"
        "  How neural embeddings fix this:\n"
        "    - Trained to place synonyms nearby in vector space\n"
        "    - Capture semantic relationships: king - man + woman ~= queen\n"
        "    - Dense: every dimension carries information\n"
        "    - GPT-style models embed entire contexts, not just words"
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_header("TOPIC 07 — Embeddings", width=66)
    print(
        "\n  An embedding converts text into a list of numbers (a vector)\n"
        "  so that mathematical operations can capture meaning.\n"
        "  Sentences with similar meaning end up with similar vectors.\n"
        "\n  This demo uses TF-IDF bag-of-words vectors — the predecessor\n"
        "  to modern neural embeddings — to show the core idea."
    )

    vocab = build_vocab(SENTENCES)
    idf   = inverse_document_frequency(SENTENCES, vocab)
    vectors = [tfidf_vector(s, vocab, idf) for s in SENTENCES]

    demo_vocabulary(vocab, SENTENCES)
    demo_tf_idf(SENTENCES, vocab, idf, vectors)
    demo_similarity_matrix(SENTENCES, vectors)
    demo_nearest_neighbour(QUERY, SENTENCES, vocab, idf, vectors)
    demo_intuition(vocab)

    print("\n" + "=" * 66)
    print("  End of Topic 07 demo.")
    print("=" * 66 + "\n")
