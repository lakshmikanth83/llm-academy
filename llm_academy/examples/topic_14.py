"""
Topic 14: Prompt Engineering
============================
Demonstrates 5 prompt patterns for the same task (summarizing a product review).
All responses are simulated — no API calls required.
"""

# ---------------------------------------------------------------------------
# Sample product review we will summarize with each prompt pattern
# ---------------------------------------------------------------------------
REVIEW = (
    "I bought this wireless keyboard three weeks ago and I am mostly satisfied. "
    "The typing feel is excellent — quiet, responsive keys with great tactile feedback. "
    "Battery life has been solid; I charge it once a week at most. "
    "Setup was painless via Bluetooth. The only downside is the keycap legends are "
    "slightly faded already, which worries me about long-term durability. "
    "Overall a good value for the price, though build quality could be better."
)


# ---------------------------------------------------------------------------
# Prompt pattern definitions
# ---------------------------------------------------------------------------
def build_prompts(review: str) -> list[dict]:
    """Return a list of prompt pattern dicts, each with name, prompt, and notes."""

    # Pattern 1 — Vague
    p1_prompt = f"Summarize this.\n\n{review}"

    # Pattern 2 — Role + task
    p2_prompt = (
        "You are a product analyst. "
        "Summarize the following customer review in exactly one sentence.\n\n"
        f"{review}"
    )

    # Pattern 3 — Structured output
    p3_prompt = (
        "Analyze the customer review below and return a JSON object with exactly "
        "three keys:\n"
        '  "summary"   : a one-sentence summary\n'
        '  "sentiment" : one of ["positive", "negative", "mixed"]\n'
        '  "score"     : an integer from 1 (worst) to 10 (best)\n\n'
        f"Review:\n{review}"
    )

    # Pattern 4 — Few-shot (3 examples before the task)
    examples = (
        "Example 1\n"
        "Review: Great headphones! Sound quality is fantastic and they are "
        "comfortable for long sessions. A bit pricey but worth it.\n"
        "Summary: High-quality, comfortable headphones that justify their premium price.\n\n"

        "Example 2\n"
        "Review: The laptop arrived with a cracked screen and customer service "
        "took two weeks to respond. Extremely disappointed.\n"
        "Summary: Product arrived damaged and support response was unacceptably slow.\n\n"

        "Example 3\n"
        "Review: Decent coffee maker. Brews quickly and the carafe keeps coffee warm. "
        "Wish the reservoir were larger.\n"
        "Summary: Fast, effective coffee maker with good heat retention but a small water tank.\n\n"
    )
    p4_prompt = (
        "Summarize each customer review in one sentence.\n\n"
        f"{examples}"
        f"Review: {review}\n"
        "Summary:"
    )

    # Pattern 5 — Chain-of-thought
    p5_prompt = (
        "Summarize the following customer review. "
        "Think step by step:\n"
        "  1) Identify the main points the reviewer raises.\n"
        "  2) Determine the overall sentiment.\n"
        "  3) Write a concise one-sentence summary.\n\n"
        f"Review:\n{review}"
    )

    return [
        {
            "id": 1,
            "name": "Vague prompt",
            "pattern": "No instruction, no context",
            "prompt": p1_prompt,
        },
        {
            "id": 2,
            "name": "Role + task",
            "pattern": "Assign a persona and specify the output format",
            "prompt": p2_prompt,
        },
        {
            "id": 3,
            "name": "Structured output",
            "pattern": "Request a specific schema (JSON) with defined fields",
            "prompt": p3_prompt,
        },
        {
            "id": 4,
            "name": "Few-shot",
            "pattern": "Provide 3 worked examples before the real task",
            "prompt": p4_prompt,
        },
        {
            "id": 5,
            "name": "Chain-of-thought (CoT)",
            "pattern": "Ask the model to reason step by step before answering",
            "prompt": p5_prompt,
        },
    ]


# ---------------------------------------------------------------------------
# Simulated model responses for each pattern
# ---------------------------------------------------------------------------
SIMULATED_RESPONSES = {
    1: (
        "The review talks about a wireless keyboard. The person likes the typing "
        "experience and battery life but is concerned about the keycap legends fading."
    ),

    2: (
        "A well-performing wireless keyboard with excellent typing feel and battery "
        "life, though early keycap fading raises concerns about long-term durability."
    ),

    3: """\
{
  "summary": "A comfortable wireless keyboard with responsive keys and solid battery life, slightly let down by early keycap wear.",
  "sentiment": "mixed",
  "score": 7
}""",

    4: (
        "A solid wireless keyboard with strong typing feel and battery life, "
        "though premature keycap fading hints at potential durability issues."
    ),

    5: """\
Step 1 — Main points:
  • Typing feel: quiet, responsive, good tactile feedback  (positive)
  • Battery life: charges once a week                      (positive)
  • Bluetooth setup: painless                              (positive)
  • Keycap legends: fading after 3 weeks                  (negative)
  • Overall value: good for the price                     (positive)
  • Build quality: could be better                        (negative)

Step 2 — Sentiment: Mixed — mostly positive with a notable durability concern.

Step 3 — Summary:
This wireless keyboard delivers an excellent typing experience and long battery
life, but early keycap fading suggests the build quality may not match its
otherwise strong performance.""",
}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def quality_notes() -> dict[int, str]:
    return {
        1: (
            "POOR — The model has no guidance. Output length, format, and focus are "
            "unpredictable. Different runs may produce wildly different results."
        ),
        2: (
            "GOOD — The persona anchors tone and expertise; the length constraint "
            "produces a focused, usable sentence every time."
        ),
        3: (
            "EXCELLENT for pipelines — The JSON schema eliminates ambiguity. "
            "Downstream code can parse the response without any text wrangling."
        ),
        4: (
            "GOOD — Examples demonstrate the expected style and length implicitly. "
            "Works even when you cannot describe the format in words."
        ),
        5: (
            "BEST for accuracy — Reasoning out loud before answering reduces "
            "hallucinations and catches nuance (mixed sentiment, caveats). "
            "Costs more tokens but improves quality on complex reviews."
        ),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
DIVIDER = "=" * 70
THIN = "-" * 70


def print_pattern(pattern: dict, response: str, note: str) -> None:
    print(DIVIDER)
    print(f"  Pattern {pattern['id']}: {pattern['name']}")
    print(f"  Technique: {pattern['pattern']}")
    print(THIN)
    print("PROMPT SENT TO MODEL:")
    print()
    # Indent the prompt for readability
    for line in pattern["prompt"].splitlines():
        print(f"  {line}")
    print()
    print("SIMULATED RESPONSE:")
    print()
    for line in response.splitlines():
        print(f"  {line}")
    print()
    print(f"ANALYSIS: {note}")
    print()


def print_before_after() -> None:
    print(DIVIDER)
    print("  BEFORE / AFTER: What prompt engineering changes")
    print(DIVIDER)
    rows = [
        ("Dimension",        "Vague (Pattern 1)",       "Best practice (Pattern 3/5)"),
        ("Output length",    "Unpredictable",           "Controlled by instruction"),
        ("Output format",    "Prose (unstructured)",    "JSON / structured steps"),
        ("Sentiment label",  "Buried in prose",         "Explicit field / reasoned step"),
        ("Score",            "Missing",                 "Numeric field 1–10"),
        ("Reproducibility",  "Low",                     "High — same prompt → same shape"),
        ("Token cost",       "Low",                     "CoT higher, others similar"),
        ("Parseability",     "Manual text parsing",     "Machine-readable (Pattern 3)"),
    ]
    col_w = [28, 26, 30]
    header, *data_rows = rows
    fmt = "  {:<{}} {:<{}} {:<{}}".format
    sep_line = "  " + "-" * (sum(col_w) + 4)

    def row_str(r):
        return fmt(r[0], col_w[0], r[1], col_w[1], r[2], col_w[2])

    print(row_str(header))
    print(sep_line)
    for r in data_rows:
        print(row_str(r))
    print()


def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 14: PROMPT ENGINEERING")
    print("  Five patterns, one task: summarize a product review")
    print(DIVIDER)
    print()
    print("PRODUCT REVIEW (input to all patterns):")
    print()
    for line in REVIEW.splitlines():
        print(f"  {line}")
    print()

    prompts = build_prompts(REVIEW)
    notes = quality_notes()

    for p in prompts:
        print_pattern(p, SIMULATED_RESPONSES[p["id"]], notes[p["id"]])

    print_before_after()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. Specificity beats brevity — a longer, well-structured prompt almost always",
        "   outperforms a short vague one.",
        "2. Role prompting anchors tone and expertise without extra tokens.",
        "3. Schema prompting (Pattern 3) is ideal when the output feeds code.",
        "4. Few-shot prompting teaches by example — useful when format is hard to describe.",
        "5. Chain-of-thought improves accuracy on nuanced tasks by making reasoning explicit.",
        "6. Combine patterns: Role + Schema + CoT is a common production recipe.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
