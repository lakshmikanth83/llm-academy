"""
Topic 06: Tokens & Tokenization
================================
Demo of whitespace/punctuation tokenization, BPE simulation,
token counts, and subword splitting.
No external dependencies — pure Python stdlib.
"""

import re
from collections import Counter, defaultdict


def print_header(title, width=64):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# PART 1: Simple whitespace + punctuation tokenizer
# ---------------------------------------------------------------------------

def simple_tokenize(text):
    """
    Split on whitespace, then isolate punctuation as separate tokens.
    Preserves contractions as single tokens (apostrophe kept inside word).
    """
    # Insert spaces around punctuation that is NOT an apostrophe inside a word
    spaced = re.sub(r"([.,!?;:\"()\[\]{}<>])", r" \1 ", text)
    tokens = spaced.split()
    return tokens


def demo_tokenizer():
    print_section("PART 1: Simple Whitespace + Punctuation Tokenizer")
    print(
        "  The simplest tokenizer splits text into words.\n"
        "  Punctuation is isolated so '.' doesn't merge with 'word.'\n"
    )

    samples = [
        "The cat sat on the mat.",
        "Hello, world! How are you?",
        "I can't stop; it's too good.",
        "LLMs (Large Language Models) are powerful.",
        "Price: $9.99 — limited time offer!",
    ]

    for text in samples:
        tokens = simple_tokenize(text)
        print(f"  Input : {text}")
        print(f"  Tokens: {tokens}")
        print(f"  Count : {len(tokens)}")
        print()


# ---------------------------------------------------------------------------
# PART 2: BPE simulation
# ---------------------------------------------------------------------------

def get_vocab_from_corpus(corpus):
    """
    Build initial character-level vocabulary from corpus.
    Each word is represented as a tuple of characters + end-of-word marker.
    Returns {word_chars_tuple: frequency}.
    """
    word_freq = Counter()
    for sentence in corpus:
        for word in sentence.lower().split():
            word = re.sub(r"[^a-z]", "", word)  # keep letters only
            if word:
                word_freq[word] += 1

    # Convert each word to a tuple of characters with </w> at end
    vocab = {}
    for word, freq in word_freq.items():
        char_tuple = tuple(list(word) + ["</w>"])
        vocab[char_tuple] = freq
    return vocab


def get_pair_stats(vocab):
    """Count occurrences of every adjacent symbol pair across the vocabulary."""
    pairs = Counter()
    for word_tuple, freq in vocab.items():
        symbols = list(word_tuple)
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair, vocab):
    """Merge all occurrences of `pair` in the vocabulary into a single symbol."""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word_tuple, freq in vocab.items():
        word_str = " ".join(word_tuple)
        merged = word_str.replace(bigram, replacement)
        new_tuple = tuple(merged.split())
        new_vocab[new_tuple] = freq
    return new_vocab


def demo_bpe():
    print_section("PART 2: Byte-Pair Encoding (BPE) Simulation")
    print(
        "  BPE starts with individual characters and iteratively merges\n"
        "  the most frequent adjacent pair into a new token.\n"
        "  This is how GPT-2/3/4 tokenizers are trained.\n"
    )

    corpus = [
        "low lower newest",
        "low wider low low",
        "newer newest lower",
        "low low new new",
        "newer lower low wide",
    ]

    print("  Corpus:")
    for s in corpus:
        print(f"    '{s}'")

    vocab = get_vocab_from_corpus(corpus)

    print("\n  Initial character-level vocabulary (word -> char tokens):")
    for word_tuple, freq in sorted(vocab.items(), key=lambda x: -x[1]):
        print(f"    {' '.join(word_tuple):<30}  freq={freq}")

    NUM_MERGES = 5
    merge_history = []

    print(f"\n  Running {NUM_MERGES} BPE merge operations...\n")
    for step in range(1, NUM_MERGES + 1):
        pairs = get_pair_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        best_count = pairs[best_pair]
        vocab = merge_pair(best_pair, vocab)
        new_token = "".join(best_pair)
        merge_history.append((step, best_pair, new_token, best_count))
        print(f"  Step {step}: merge {best_pair}  ->  '{new_token}'  (appeared {best_count} times)")

    print("\n  Vocabulary after all merges:")
    for word_tuple, freq in sorted(vocab.items(), key=lambda x: -x[1]):
        print(f"    {' '.join(word_tuple):<30}  freq={freq}")

    print("\n  Summary of learned merge rules:")
    print(f"  {'Step':<6} {'Merged pair':<20} {'New token':<14} {'Frequency'}")
    print(f"  {'-'*4:<6} {'-'*18:<20} {'-'*9:<14} {'-'*9}")
    for step, pair, token, count in merge_history:
        pair_str = f"'{pair[0]}' + '{pair[1]}'"
        print(f"  {step:<6} {pair_str:<20} '{token}'{'':6} {count}")

    print(
        "\n  Key insight: BPE learns that frequent character sequences\n"
        "  should become single tokens. 'low' appears often, so 'l'+'o'\n"
        "  gets merged first, then 'lo'+'w', etc."
    )


# ---------------------------------------------------------------------------
# PART 3: Token counts for different texts
# ---------------------------------------------------------------------------

def count_tokens_approx(text):
    """
    Approximate GPT-style token count heuristic:
    ~4 characters per token on average for English.
    Also returns the whitespace token count for comparison.
    """
    word_count   = len(text.split())
    char_count   = len(text.replace(" ", ""))
    approx_bpe   = max(1, round(len(text) / 4))
    return word_count, char_count, approx_bpe


def demo_token_counts():
    print_section("PART 3: Token Counts for Different Text Types")
    print(
        "  Different types of text tokenize very differently.\n"
        "  Code and rare words use more tokens per character.\n"
    )

    samples = [
        ("Plain English",     "The quick brown fox jumps over the lazy dog."),
        ("Technical jargon",  "The transformer architecture uses multi-head self-attention mechanisms."),
        ("Python code",       "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
        ("Mixed numbers",     "In 2024, the model scored 98.7% on 1,234 benchmark tasks."),
        ("Long rare word",    "Supercalifragilisticexpialidocious antidisestablishmentarianism."),
        ("Emoji/Unicode",     "Hello world 🌍 — café résumé naïve"),
    ]

    col_w = [18, 8, 8, 12]
    print(f"  {'Text type':<{col_w[0]}} {'Words':<{col_w[1]}} {'Chars':<{col_w[2]}} {'~BPE tokens':<{col_w[3]}} Sample")
    print(f"  {'-'*16:<{col_w[0]}} {'-'*5:<{col_w[1]}} {'-'*5:<{col_w[2]}} {'-'*10:<{col_w[3]}} {'-'*35}")
    for label, text in samples:
        wc, cc, bpe = count_tokens_approx(text)
        display = text[:40].replace("\n", "\\n")
        if len(text) > 40:
            display += "..."
        print(f"  {label:<{col_w[0]}} {wc:<{col_w[1]}} {cc:<{col_w[2]}} {bpe:<{col_w[3]}} {display}")


# ---------------------------------------------------------------------------
# PART 4: Subword splitting examples
# ---------------------------------------------------------------------------

SUBWORD_RULES = {
    # prefix  -> keep prefix as token, recurse on remainder
    "un":     True,
    "re":     True,
    "pre":    True,
    "dis":    True,
    "over":   True,
    "under":  True,
    "out":    True,
    # suffix -> split these from the end
    "ness":   True,
    "tion":   True,
    "ment":   True,
    "able":   True,
    "ible":   True,
    "ful":    True,
    "less":   True,
    "ing":    True,
    "ed":     True,
    "er":     True,
    "est":    True,
    "ly":     True,
    "ize":    True,
    "ise":    True,
    "ous":    True,
    "ive":    True,
    "al":     True,
}

PREFIXES = ["un", "re", "pre", "dis", "over", "under", "out"]
SUFFIXES = ["ness", "tion", "ment", "able", "ible", "ful", "less",
            "ing", "ed", "er", "est", "ly", "ize", "ise", "ous", "ive", "al"]


def naive_subword_split(word):
    """
    Rule-based subword splitter (approximates what BPE learns).
    Strips known prefixes then known suffixes, keeping the stem.
    """
    tokens = []
    remaining = word.lower()

    # Strip prefix
    for pfx in PREFIXES:
        if remaining.startswith(pfx) and len(remaining) > len(pfx) + 2:
            tokens.append(pfx)
            remaining = remaining[len(pfx):]
            break

    # Strip suffix (one pass, longest first)
    found_suffix = None
    for sfx in sorted(SUFFIXES, key=len, reverse=True):
        if remaining.endswith(sfx) and len(remaining) > len(sfx) + 1:
            found_suffix = sfx
            remaining = remaining[: -len(sfx)]
            break

    tokens.append(remaining)
    if found_suffix:
        tokens.append(found_suffix)

    return tokens


def demo_subword():
    print_section("PART 4: Subword Tokenization Examples")
    print(
        "  Real tokenizers (BPE, WordPiece) often split rare or long words\n"
        "  into recognisable subword pieces rather than treating the whole\n"
        "  word as a single token or falling back to characters.\n"
        "\n  Why? Common morphemes ('un-', '-ness', '-ing') appear across\n"
        "  many words, so the model re-uses learned representations for them.\n"
    )

    words = [
        "unhappiness",
        "tokenization",
        "preprocessing",
        "overestimated",
        "disestablishment",
        "underdeveloped",
        "outperforming",
        "reusability",
        "transformative",
        "generalization",
    ]

    print(f"  {'Word':<22} Subword pieces")
    print(f"  {'-'*20:<22} {'-'*30}")
    for word in words:
        pieces = naive_subword_split(word)
        pieces_str = " + ".join(f'"{p}"' for p in pieces)
        print(f"  {word:<22} [{pieces_str}]")

    print(
        "\n  Note: This is a rule-based approximation to show the concept.\n"
        "  Real BPE splits are learned from data and look slightly different,\n"
        "  e.g. GPT-2 tokenizes 'unhappiness' as ['un', 'happ', 'iness']."
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_header("TOPIC 06 — Tokens & Tokenization", width=64)
    print(
        "\n  Tokenization converts raw text into a sequence of integer IDs\n"
        "  that a language model can process. The choice of tokenizer\n"
        "  affects vocabulary size, model efficiency, and what the\n"
        "  model finds easy or hard to learn.\n"
        "\n  Four concepts are demonstrated:\n"
        "    1. Simple whitespace + punctuation tokenizer\n"
        "    2. Byte-Pair Encoding (BPE) — how tokenizers are trained\n"
        "    3. Token counts across different text types\n"
        "    4. Subword splitting for morphologically rich words"
    )

    demo_tokenizer()
    demo_bpe()
    demo_token_counts()
    demo_subword()

    print("\n" + "=" * 64)
    print("  End of Topic 06 demo.")
    print("=" * 64 + "\n")
