"""
Topic 15: Zero-shot, One-shot, Few-shot
========================================
Runs the SAME intent-classification task through zero-shot, one-shot, and
few-shot prompts and measures how accuracy and output-format consistency
change as demonstrations are added. The LLM is simulated (deterministic,
rule-based) so the whole demo runs offline with no API key.
"""

from __future__ import annotations

import textwrap

DIVIDER = "=" * 70
THIN = "-" * 70
WRAP_W = 66


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=WRAP_W, initial_indent=indent,
                         subsequent_indent=indent)


# ---------------------------------------------------------------------------
# THE TASK: classify a customer message into one of three support intents
# ---------------------------------------------------------------------------

VALID_LABELS = ["refund_request", "shipping_question", "general_inquiry"]

INSTRUCTION = (
    "Classify the customer support message below into exactly one of these "
    "categories: refund_request, shipping_question, general_inquiry. "
    "Respond with only the category label."
)

# The same 8 messages are run through every prompt style so the comparison
# is fair. (message, ground_truth_label)
TEST_SET = [
    ("I want my money back for this broken item.", "refund_request"),
    ("When will my package arrive?", "shipping_question"),
    ("Do you have this in blue?", "general_inquiry"),
    ("This product broke after one day, I need a refund.", "refund_request"),
    ("My order hasn't shipped yet, what's the status?", "shipping_question"),
    ("What are your store hours?", "general_inquiry"),
    ("I'm not happy with the item, can I return it for a refund?", "refund_request"),
    ("The tracking number isn't working, where is my order?", "shipping_question"),
]

# Demonstrations used to build one-shot / few-shot prompts. These are
# deliberately DIFFERENT messages from the test set — a real few-shot
# prompt teaches a pattern, it does not leak the test answers.
ONE_SHOT_EXAMPLE = {
    "message": "My package shows delivered but I never received it.",
    "label": "shipping_question",
}

FEW_SHOT_EXAMPLES = [
    {"message": "The item I received is damaged and I'd like my money back.",
     "label": "refund_request"},
    {"message": "My package shows delivered but I never received it.",
     "label": "shipping_question"},
    {"message": "Can you tell me if you ship internationally?",
     "label": "general_inquiry"},
]


# ---------------------------------------------------------------------------
# PROMPT BUILDERS
# ---------------------------------------------------------------------------

def build_zero_shot_prompt(message: str) -> str:
    """Instruction only — no demonstrations of the desired input/output pattern."""
    return f'{INSTRUCTION}\n\nMessage: "{message}"\nCategory:'


def build_one_shot_prompt(message: str) -> str:
    """Instruction + exactly one demonstration."""
    ex = ONE_SHOT_EXAMPLE
    return (
        f"{INSTRUCTION}\n\n"
        "Example:\n"
        f'Message: "{ex["message"]}"\n'
        f"Category: {ex['label']}\n\n"
        f'Message: "{message}"\n'
        "Category:"
    )


def build_few_shot_prompt(message: str) -> str:
    """Instruction + several demonstrations, one per category."""
    blocks = []
    for ex in FEW_SHOT_EXAMPLES:
        blocks.append(f'Message: "{ex["message"]}"\nCategory: {ex["label"]}')
    examples_text = "\n\n".join(blocks)
    return (
        f"{INSTRUCTION}\n\n"
        f"{examples_text}\n\n"
        f'Message: "{message}"\n'
        "Category:"
    )


def token_estimate(text: str) -> int:
    """Rough word-based token estimate (~1.3 tokens/word) — no tokenizer needed."""
    return round(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# SIMULATED LLM — deterministic responses, keyed by (message, style)
# ---------------------------------------------------------------------------
#
# These responses model a real, well-documented phenomenon: without examples
# a model may (a) still get easy cases right from prior knowledge, but
# (b) drift into free-form phrasing or invent a plausible-but-wrong label for
# ambiguous/novel cases, because it was never shown the *exact* output
# contract. Each example demonstration nudges the model toward the right
# format and, eventually, the right disambiguation.

ZERO_SHOT_RESPONSES = {
    "I want my money back for this broken item.": "refund_request",
    "When will my package arrive?": "shipping_question",
    "Do you have this in blue?":
        "This sounds like a question about product availability and color options.",
    "This product broke after one day, I need a refund.": "product_return",
    "My order hasn't shipped yet, what's the status?": "shipping_question",
    "What are your store hours?": "general_inquiry",
    "I'm not happy with the item, can I return it for a refund?": "return_request",
    "The tracking number isn't working, where is my order?": "shipping_question",
}

ONE_SHOT_RESPONSES = {
    "I want my money back for this broken item.": "refund_request",
    "When will my package arrive?": "shipping_question",
    "Do you have this in blue?": "general_inquiry",
    "This product broke after one day, I need a refund.": "refund_request",
    "My order hasn't shipped yet, what's the status?": "shipping_question",
    "What are your store hours?": "general_inquiry",
    # Only one demonstration was shown, and it was a shipping example, so an
    # ambiguous "return"-flavoured refund message drifts toward shipping.
    "I'm not happy with the item, can I return it for a refund?": "shipping_question",
    "The tracking number isn't working, where is my order?": "shipping_question",
}

FEW_SHOT_RESPONSES = {
    "I want my money back for this broken item.": "refund_request",
    "When will my package arrive?": "shipping_question",
    "Do you have this in blue?": "general_inquiry",
    "This product broke after one day, I need a refund.": "refund_request",
    "My order hasn't shipped yet, what's the status?": "shipping_question",
    "What are your store hours?": "general_inquiry",
    "I'm not happy with the item, can I return it for a refund?": "refund_request",
    "The tracking number isn't working, where is my order?": "shipping_question",
}


# ---------------------------------------------------------------------------
# GRADING — parse the model's raw text into a label, or flag format drift
# ---------------------------------------------------------------------------

def normalize(raw: str) -> str:
    text = raw.strip().lower().rstrip(".!")
    return text.replace(" ", "_").replace("-", "_")


def parse_label(raw: str) -> str | None:
    """Return the label if the raw output is a clean match, else None."""
    norm = normalize(raw)
    return norm if norm in VALID_LABELS else None


def grade(responses: dict) -> dict:
    """Score a style's responses against ground truth. Returns a stats dict."""
    rows = []
    correct = 0
    format_failures = 0
    for message, truth in TEST_SET:
        raw = responses[message]
        parsed = parse_label(raw)
        is_format_ok = parsed is not None
        is_correct = parsed == truth
        if is_correct:
            correct += 1
        if not is_format_ok:
            format_failures += 1
        rows.append({
            "message": message, "truth": truth, "raw": raw,
            "parsed": parsed, "correct": is_correct, "format_ok": is_format_ok,
        })
    total = len(TEST_SET)
    return {
        "rows": rows,
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "format_failures": format_failures,
    }


# ---------------------------------------------------------------------------
# DEMO RUNNER — one per prompt style
# ---------------------------------------------------------------------------

STYLES = [
    {
        "name": "ZERO-SHOT",
        "n_examples": 0,
        "builder": build_zero_shot_prompt,
        "responses": ZERO_SHOT_RESPONSES,
    },
    {
        "name": "ONE-SHOT",
        "n_examples": 1,
        "builder": build_one_shot_prompt,
        "responses": ONE_SHOT_RESPONSES,
    },
    {
        "name": "FEW-SHOT (3 examples)",
        "n_examples": 3,
        "builder": build_few_shot_prompt,
        "responses": FEW_SHOT_RESPONSES,
    },
]


def run_style_demo(style: dict) -> dict:
    name = style["name"]
    builder = style["builder"]
    responses = style["responses"]

    print(DIVIDER)
    print(f"  {name}  —  {style['n_examples']} demonstration(s) in the prompt")
    print(DIVIDER)

    # Show one representative full prompt (the trickiest test message,
    # where the styles start to disagree) so the reader sees exactly what
    # the model receives.
    sample_message = TEST_SET[6][0]
    sample_prompt = builder(sample_message)
    print()
    print("  Sample constructed prompt:")
    print(THIN)
    for line in sample_prompt.splitlines():
        print(f"  {line}")
    print(THIN)
    print(f"  (~{token_estimate(sample_prompt)} tokens)")

    stats = grade(responses)

    print()
    print("  Results across the test set:")
    print(THIN)
    print(f"  {'#':<3}{'Message':<42}{'Truth':<18}{'Model output':<22}Result")
    print(f"  {'-'*2:<3}{'-'*38:<42}{'-'*16:<18}{'-'*20:<22}{'-'*6}")
    for i, row in enumerate(stats["rows"], start=1):
        msg_short = row["message"] if len(row["message"]) <= 39 else row["message"][:36] + "..."
        if row["correct"]:
            mark = "OK"
        elif not row["format_ok"]:
            mark = "FORMAT DRIFT"
        else:
            mark = "WRONG"
        print(f"  {i:<3}{msg_short:<42}{row['truth']:<18}{row['raw'][:20]:<22}{mark}")

    print()
    print(
        f"  Accuracy: {stats['correct']}/{stats['total']} "
        f"({stats['accuracy']*100:.1f}%)   |   "
        f"Format drift (unparsable label): {stats['format_failures']}/{stats['total']}"
    )
    print()

    return {"name": name, "stats": stats, "prompt_tokens": token_estimate(sample_prompt),
            "n_examples": style["n_examples"]}


# ---------------------------------------------------------------------------
# COMPARISON TABLE
# ---------------------------------------------------------------------------

def demo_comparison(results: list) -> None:
    print(DIVIDER)
    print("  COMPARISON: Zero-shot vs One-shot vs Few-shot")
    print(DIVIDER)
    print()

    by_name = {r["name"]: r for r in results}
    zero = by_name["ZERO-SHOT"]
    one = by_name["ONE-SHOT"]
    few = by_name["FEW-SHOT (3 examples)"]

    def acc(r):
        return f"{r['stats']['accuracy']*100:.0f}%"

    def latency_tag(n_examples):
        return {0: "Lowest", 1: "Medium"}.get(n_examples, "Highest")

    rows = [
        ("Examples in prompt", "0", "1", "3"),
        ("Accuracy (this test set)", acc(zero), acc(one), acc(few)),
        ("Format-drift errors", str(zero["stats"]["format_failures"]),
         str(one["stats"]["format_failures"]), str(few["stats"]["format_failures"])),
        ("Approx. prompt tokens", str(zero["prompt_tokens"]),
         str(one["prompt_tokens"]), str(few["prompt_tokens"])),
        ("Relative token cost", "Lowest", "Medium", "Highest"),
        ("Relative latency", latency_tag(0), latency_tag(1), latency_tag(3)),
        ("Best for", "Well-known tasks", "Style w/ 1 sample", "Custom/novel labels"),
    ]

    col1, col2, col3, col4 = 24, 16, 16, 16
    print(f"  {'Aspect':<{col1}} {'Zero-shot':<{col2}} {'One-shot':<{col3}} {'Few-shot':<{col4}}")
    print("  " + "-" * (col1 + col2 + col3 + col4))
    for aspect, z, o, f in rows:
        print(f"  {aspect:<{col1}} {z:<{col2}} {o:<{col3}} {f:<{col4}}")

    print(
        "\n  Reading the numbers:\n"
        f"    Zero-shot got the EASY cases right but drifted into free text or\n"
        f"    invented labels ('product_return', 'return_request') on messages\n"
        f"    it hadn't seen a pattern for — that's a real failure mode, not a\n"
        f"    contrived one: the model had no contract for the exact output.\n"
        f"    One example fixed the output FORMAT everywhere, but one example\n"
        f"    can't disambiguate every category, so one confusion remained.\n"
        f"    Three examples (one per category) gave the model a demonstration\n"
        f"    of every label it needed, and it went 8/8 on this test set."
    )
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 15: ZERO-SHOT, ONE-SHOT, FEW-SHOT PROMPTING")
    print(DIVIDER)
    print()
    print("  Task: classify customer support messages into one of three intents:")
    print(f"    {', '.join(VALID_LABELS)}")
    print()
    print(
        "  The SAME 8-message test set is run through three prompt styles.\n"
        "  Only the prompt changes — the (simulated) model, the task, and the\n"
        "  test messages are held constant, so any accuracy difference comes\n"
        "  purely from how many demonstrations were included."
    )
    print()

    results = []
    for style in STYLES:
        results.append(run_style_demo(style))

    demo_comparison(results)

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. Zero-shot relies entirely on the model's prior knowledge of the task —",
        "   it can drift into free-form text or invented labels on anything novel.",
        "2. One example is usually enough to lock in the OUTPUT FORMAT, even if",
        "   it can't teach every category the model needs to disambiguate.",
        "3. Few-shot (one demonstration per class) is the most reliable way to",
        "   teach a custom or unusual label scheme — it directly showed the fix.",
        "4. More examples cost more tokens and latency — the accuracy gain must",
        "   be worth the extra prompt size for your use case.",
        "5. This is in-context learning (ICL): the model adapts from examples in",
        "   the prompt with zero weight updates — nothing was fine-tuned here.",
        "6. Quality and category coverage beat raw example count — 3 well-chosen",
        "   demonstrations outperformed 0, and would likely beat 20 redundant ones.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
