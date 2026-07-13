"""
Topic 12: Temperature & Sampling
==================================
Demo of temperature scaling on a toy next-word probability distribution.
No external dependencies — pure Python stdlib.
"""

import math
import random
from collections import Counter


def print_header(title, width=66):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# PROBABILITY & TEMPERATURE UTILITIES
# ---------------------------------------------------------------------------

# Raw logits representing the model's unnormalised scores for each word.
# These are chosen to produce the desired probabilities after softmax.
BASE_LOGITS = {
    "good":     0.916,   # -> ~50% at T=1
    "great":    0.405,   # -> ~30% at T=1
    "bad":     -0.916,   # -> ~10% at T=1
    "terrible": -0.916,  # -> ~10% at T=1
}


def softmax(logit_dict, temperature):
    """
    Apply temperature scaling then softmax:
      scaled_logit = logit / temperature
      prob[i] = exp(scaled[i]) / sum(exp(scaled))

    Temperature < 1 -> sharper (more confident) distribution
    Temperature = 1 -> original distribution
    Temperature > 1 -> flatter (more random) distribution
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    scaled = {word: logit / temperature for word, logit in logit_dict.items()}

    # Numerical stability: subtract max before exp
    max_val = max(scaled.values())
    exps    = {word: math.exp(v - max_val) for word, v in scaled.items()}
    total   = sum(exps.values())
    return {word: exp_val / total for word, exp_val in exps.items()}


def sample_from_distribution(prob_dict, rng):
    """
    Sample one word according to probabilities using inverse CDF.
    Uses a provided random.Random instance for reproducibility.
    """
    words = list(prob_dict.keys())
    probs = [prob_dict[w] for w in words]
    r = rng.random()
    cumulative = 0.0
    for word, p in zip(words, probs):
        cumulative += p
        if r <= cumulative:
            return word
    return words[-1]  # fallback for floating-point edge cases


def distribution_bar(prob_dict, width=30):
    """Return a dict of word -> ASCII bar string for display."""
    bars = {}
    for word, p in prob_dict.items():
        filled = int(p * width)
        bar = "█" * filled + "░" * (width - filled)
        bars[word] = bar
    return bars


# ---------------------------------------------------------------------------
# DEMO SECTIONS
# ---------------------------------------------------------------------------

def demo_base_distribution():
    print_section("BASE DISTRIBUTION (Temperature = 1.0)")
    print(
        "  The model assigns these probabilities to the next word\n"
        "  after 'The food was really ___':\n"
    )
    probs = softmax(BASE_LOGITS, temperature=1.0)
    bars  = distribution_bar(probs)
    print(f"  {'Word':<12} {'Probability':<12} Distribution")
    print(f"  {'-'*10:<12} {'-'*10:<12} {'-'*32}")
    for word, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {word:<12} {p:>7.1%}     {bars[word]}")


def demo_temperature_effect():
    print_section("HOW TEMPERATURE CHANGES THE DISTRIBUTION")
    print(
        "  Temperature T divides every logit before softmax.\n"
        "    T < 1  ->  logits spread further apart -> winner wins bigger\n"
        "    T = 1  ->  logits unchanged -> original probabilities\n"
        "    T > 1  ->  logits compressed toward zero -> probs flatten out\n"
    )

    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
    bar_width = 28

    print(f"\n  {'Word':<12}", end="")
    for t in temperatures:
        label = f"T={t}"
        print(f"  {label:<10}", end="")
    print()
    print(f"  {'-'*10:<12}", end="")
    for _ in temperatures:
        print(f"  {'-'*8:<10}", end="")
    print()

    for word in sorted(BASE_LOGITS.keys()):
        print(f"  {word:<12}", end="")
        for t in temperatures:
            probs = softmax(BASE_LOGITS, t)
            p = probs[word]
            print(f"  {p:>7.1%}   ", end="")
        print()

    print(
        "\n  Key observations:\n"
        "    T=0.1 : 'good' gets ~100% — the model always picks the top word.\n"
        "    T=1.0 : original distribution — 50/30/10/10.\n"
        "    T=2.0 : 'terrible' rises from 10% to ~22% — more surprising outputs."
    )


def demo_sampling(n_samples=20, seed=42):
    print_section(f"SAMPLING {n_samples} WORDS AT DIFFERENT TEMPERATURES")
    print(
        "  Stochastic sampling: even a 10% word will occasionally be chosen.\n"
        "  Higher temperature -> rare words appear more often.\n"
    )

    temperatures = [0.1, 1.0, 2.0]

    for temp in temperatures:
        rng   = random.Random(seed)
        probs = softmax(BASE_LOGITS, temp)
        samples = [sample_from_distribution(probs, rng) for _ in range(n_samples)]
        counts  = Counter(samples)

        print(f"\n  Temperature = {temp}")
        print(f"  Probabilities: " +
              "  ".join(f"{w}={probs[w]:.1%}" for w in sorted(BASE_LOGITS)))
        print(f"  Samples:  {' '.join(samples)}")
        print(f"  Frequency counts:")
        for word in sorted(BASE_LOGITS.keys()):
            count   = counts.get(word, 0)
            pct     = count / n_samples
            bar     = "█" * count + "░" * (n_samples - count)
            print(f"    {word:<12} {count:>3}/{n_samples}  {pct:>5.0%}  {bar}")


def demo_greedy_vs_sampling():
    print_section("GREEDY DECODING vs SAMPLING")
    print(
        "  Greedy decoding (T -> 0) always picks the highest-probability\n"
        "  word. This is deterministic but can be repetitive or bland.\n"
        "\n  Sampling (T >= 1) introduces randomness, allowing the model\n"
        "  to produce varied, creative outputs.\n"
    )

    sentence_start = "The food was really"
    words_greedy   = []
    words_sampled  = []
    rng            = random.Random(0)

    for step in range(6):
        probs_greedy  = softmax(BASE_LOGITS, temperature=0.01)
        probs_sampled = softmax(BASE_LOGITS, temperature=1.2)
        words_greedy.append(max(probs_greedy,  key=probs_greedy.get))
        words_sampled.append(sample_from_distribution(probs_sampled, rng))

    print(f"  Prompt: '{sentence_start} ___'\n")
    print(f"  Greedy  (T=0.01): '{sentence_start} {' '.join(words_greedy)}'")
    print(
        "    -> Same word every time. Repetitive but predictable.\n"
    )
    print(f"  Sampled (T=1.20): '{sentence_start} {' '.join(words_sampled)}'")
    print(
        "    -> Different words chosen. More natural variation."
    )


def demo_temperature_guide():
    print_section("PRACTICAL TEMPERATURE GUIDE")
    rows = [
        ("0.0 – 0.3", "Near-deterministic",  "Code, math, factual Q&A"),
        ("0.4 – 0.7", "Balanced",             "Summarisation, translation"),
        ("0.8 – 1.0", "Creative",             "Story writing, brainstorming"),
        ("1.1 – 1.5", "Very random",          "Ideation, novelty exploration"),
        ("2.0+",      "Near-uniform",          "Testing / debugging tokenizer"),
    ]
    print(f"  {'Range':<14} {'Character':<20} Typical use case")
    print(f"  {'-'*12:<14} {'-'*18:<20} {'-'*30}")
    for rng, char, use in rows:
        print(f"  {rng:<14} {char:<20} {use}")

    print(
        "\n  Note: temperature is not the only sampling parameter.\n"
        "  Real APIs also offer:\n"
        "    top_p   (nucleus sampling) — sample from top-P probability mass\n"
        "    top_k   — restrict to top-K words before sampling\n"
        "  These are often used together with temperature."
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_header("TOPIC 12 — Temperature & Sampling", width=66)
    print(
        "\n  When a language model generates text it does not simply pick\n"
        "  the single most likely next word every time — that would be\n"
        "  repetitive and boring. Instead it samples from a probability\n"
        "  distribution, and *temperature* controls how sharp or flat\n"
        "  that distribution is.\n"
        "\n  This demo uses a toy 4-word distribution:\n"
        "    'good' 50%  |  'great' 30%  |  'bad' 10%  |  'terrible' 10%"
    )

    demo_base_distribution()
    demo_temperature_effect()
    demo_sampling(n_samples=20, seed=42)
    demo_greedy_vs_sampling()
    demo_temperature_guide()

    print("\n" + "=" * 66)
    print("  End of Topic 12 demo.")
    print("=" * 66 + "\n")
