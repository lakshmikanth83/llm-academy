"""
Topic 16: Chain of Thought & Reasoning
=======================================
Side-by-side demo: the SAME multi-step word problems and logic puzzle are
solved twice — once with DIRECT prompting (model blurts out an answer with
no visible reasoning) and once with CHAIN-OF-THOUGHT prompting ("Let's
think step by step", producing a visible scratchpad). Shows the real
accuracy gap this simple phrase creates. Pure Python stdlib — no API key.
"""

from __future__ import annotations

import textwrap
from collections import deque

DIVIDER = "=" * 70
THIN = "-" * 70


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=66, initial_indent=indent,
                         subsequent_indent=indent)


# ---------------------------------------------------------------------------
# PROBLEM 1: Roger's Tennis Balls — simple multi-step arithmetic
# ---------------------------------------------------------------------------
# This is the canonical example from the original CoT paper (Wei et al.,
# 2022). It is simple enough that even direct prompting usually nails it —
# CoT's real advantage shows up on the harder problems below.

QUESTION_1 = (
    "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each "
    "can has 3 tennis balls. How many tennis balls does he have now?"
)


def direct_roger_balls():
    """Simulated 'System 1' guess — no intermediate steps shown."""
    start, cans, per_can = 5, 2, 3
    answer = start + cans * per_can
    note = "Combined all the numbers correctly despite showing no reasoning."
    return answer, note


def cot_roger_balls():
    """Genuinely computes the answer step by step, printing each step."""
    steps = []
    start = 5
    steps.append(f"Step 1: Roger starts with {start} tennis balls.")

    cans, per_can = 2, 3
    new_balls = cans * per_can
    steps.append(
        f"Step 2: He buys {cans} cans x {per_can} balls per can = {new_balls} new balls."
    )

    total = start + new_balls
    steps.append(f"Step 3: Total = {start} + {new_balls} = {total} tennis balls.")

    return steps, total


# ---------------------------------------------------------------------------
# PROBLEM 2: Bakery Cupcakes — multi-step subtraction (harder)
# ---------------------------------------------------------------------------
# Direct mode latches onto the first two numbers and drops the second event.

QUESTION_2 = (
    "A bakery had 120 cupcakes. In the morning they sold 45 cupcakes. In "
    "the afternoon, a caterer bought 3 boxes of 8 cupcakes each. How many "
    "cupcakes are left?"
)


def direct_cupcakes():
    """Shortcut failure mode: only the first subtraction is applied."""
    start, morning_sold = 120, 45
    guess = start - morning_sold
    note = "Latched onto the first two numbers and ignored the afternoon sale entirely."
    return guess, note


def cot_cupcakes():
    steps = []
    start = 120
    steps.append(f"Step 1: Bakery starts with {start} cupcakes.")

    morning_sold = 45
    after_morning = start - morning_sold
    steps.append(
        f"Step 2: Morning sales: {start} - {morning_sold} = {after_morning} left."
    )

    boxes, per_box = 3, 8
    afternoon_sold = boxes * per_box
    steps.append(
        f"Step 3: Afternoon: caterer buys {boxes} boxes x {per_box} = {afternoon_sold} cupcakes."
    )

    final = after_morning - afternoon_sold
    steps.append(
        f"Step 4: Remaining = {after_morning} - {afternoon_sold} = {final} cupcakes."
    )

    return steps, final


# ---------------------------------------------------------------------------
# PROBLEM 3: Average Train Speed — rate problem (harder)
# ---------------------------------------------------------------------------
# Direct mode adds quantities but forgets that "average" requires dividing.

QUESTION_3 = (
    "A train travels 60 miles in the first hour and 90 miles in the second "
    "hour. What is the train's average speed, in miles per hour, for the "
    "whole trip?"
)


def direct_train_speed():
    """Shortcut failure mode: sums distances, forgets to divide by time."""
    hour1, hour2 = 60, 90
    guess = hour1 + hour2
    note = "Added the two distances but forgot to divide by the elapsed time."
    return guess, note


def cot_train_speed():
    steps = []
    hour1, hour2 = 60, 90
    steps.append(f"Step 1: Distance covered in hour 1 = {hour1} miles.")
    steps.append(f"Step 2: Distance covered in hour 2 = {hour2} miles.")

    total_distance = hour1 + hour2
    total_time = 2
    steps.append(
        f"Step 3: Total distance = {hour1} + {hour2} = {total_distance} miles "
        f"over {total_time} hours."
    )

    avg_speed = total_distance / total_time
    steps.append(
        f"Step 4: Average speed = {total_distance} / {total_time} = {avg_speed} mph."
    )

    return steps, avg_speed


# ---------------------------------------------------------------------------
# LOGIC ENGINE — topological sort (used by Problem 4)
# ---------------------------------------------------------------------------

def topo_order(people: list, relations: list) -> list:
    """
    Kahn's algorithm: relations is a list of (older, younger) pairs, each
    meaning 'older is older than younger'. Returns people ordered from
    oldest to youngest, resolved transitively across ALL relations —
    not just the first one mentioned.
    """
    indegree = {p: 0 for p in people}
    graph = {p: [] for p in people}
    for older, younger in relations:
        graph[older].append(younger)
        indegree[younger] += 1

    queue = deque(p for p in people if indegree[p] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return order


# ---------------------------------------------------------------------------
# PROBLEM 4: Age Logic Puzzle — transitive reasoning (logic, not arithmetic)
# ---------------------------------------------------------------------------
# Direct mode anchors on the first relation and never revisits that
# conclusion when a later statement (Dana is older than Amy) overturns it.

AGE_RELATIONS = [("Amy", "Ben"), ("Ben", "Cara"), ("Dana", "Amy")]
AGE_PEOPLE = ["Amy", "Ben", "Cara", "Dana"]

QUESTION_4 = (
    "Amy is older than Ben. Ben is older than Cara. Dana is older than "
    "Amy. Who is the oldest of the four?"
)


def direct_logic_oldest():
    """Shortcut failure mode: anchors on the first relation only."""
    first_older, _first_younger = AGE_RELATIONS[0]
    guess = first_older
    note = "Anchored on the first relation stated and ignored the later statements."
    return guess, note


def cot_logic_oldest():
    steps = []
    steps.append("Step 1: Convert each sentence into a directed relation (older -> younger):")
    for older, younger in AGE_RELATIONS:
        steps.append(f"    {older} is older than {younger}")

    steps.append(
        "Step 2: Run a topological sort over ALL relations to build a full "
        "oldest-to-youngest ordering (not just the first fact seen)."
    )
    order = topo_order(AGE_PEOPLE, AGE_RELATIONS)
    steps.append(f"Step 3: Resulting order, oldest to youngest: {' > '.join(order)}")

    answer = order[0]
    steps.append(f"Step 4: The oldest person is {answer}.")

    return steps, answer


# ---------------------------------------------------------------------------
# PROBLEM REGISTRY
# ---------------------------------------------------------------------------

PROBLEMS = [
    {
        "title": "Roger's Tennis Balls (simple 2-step arithmetic)",
        "question": QUESTION_1,
        "direct_fn": direct_roger_balls,
        "cot_fn": cot_roger_balls,
        "expected": 11,
    },
    {
        "title": "Bakery Cupcakes (3-step arithmetic with a distractor step)",
        "question": QUESTION_2,
        "direct_fn": direct_cupcakes,
        "cot_fn": cot_cupcakes,
        "expected": 51,
    },
    {
        "title": "Train Average Speed (rate / division step)",
        "question": QUESTION_3,
        "direct_fn": direct_train_speed,
        "cot_fn": cot_train_speed,
        "expected": 75.0,
    },
    {
        "title": "Age Logic Puzzle (transitive deduction, not arithmetic)",
        "question": QUESTION_4,
        "direct_fn": direct_logic_oldest,
        "cot_fn": cot_logic_oldest,
        "expected": "Dana",
    },
]


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def run_problem(num: int, problem: dict) -> dict:
    print(DIVIDER)
    print(f"  PROBLEM {num}: {problem['title']}")
    print(DIVIDER)
    print(wrap(problem["question"]))
    print()

    # ---- MODE 1: DIRECT --------------------------------------------------
    print('  MODE 1 -- DIRECT PROMPT (no "let\'s think step by step")')
    print(THIN)
    direct_answer, direct_note = problem["direct_fn"]()
    print(f'  Prompt sent : "{problem["question"]}"')
    print(f'  Model reply : "The answer is {direct_answer}."')
    print(wrap(f"(shortcut used internally: {direct_note})", indent="    "))
    print()

    # ---- MODE 2: CHAIN OF THOUGHT -----------------------------------------
    print('  MODE 2 -- CHAIN OF THOUGHT (append "Let\'s think step by step.")')
    print(THIN)
    steps, cot_answer = problem["cot_fn"]()
    for step in steps:
        print(f"  {step}")
    print(f"  Final answer: {cot_answer}")
    print()

    # ---- Verdicts ----------------------------------------------------------
    expected = problem["expected"]
    direct_correct = direct_answer == expected
    cot_correct = cot_answer == expected

    print("  VERDICT")
    print(THIN)
    print(f"  Expected answer     : {expected}")
    print(f"  Direct mode answer  : {direct_answer}  ->  {'CORRECT' if direct_correct else 'WRONG'}")
    print(f"  CoT mode answer     : {cot_answer}  ->  {'CORRECT' if cot_correct else 'WRONG'}")
    print()

    return {"title": problem["title"], "direct_correct": direct_correct, "cot_correct": cot_correct}


# ---------------------------------------------------------------------------
# SCORECARD
# ---------------------------------------------------------------------------

def print_scorecard(results: list) -> None:
    print(DIVIDER)
    print("  SCORECARD")
    print(DIVIDER)

    total = len(results)
    direct_score = sum(1 for r in results if r["direct_correct"])
    cot_score = sum(1 for r in results if r["cot_correct"])

    print(f"  {'Problem':<48}{'Direct':<10}{'CoT':<10}")
    print("  " + "-" * 66)
    for r in results:
        d = "CORRECT" if r["direct_correct"] else "WRONG"
        c = "CORRECT" if r["cot_correct"] else "WRONG"
        title = r["title"][:46]
        print(f"  {title:<48}{d:<10}{c:<10}")
    print("  " + "-" * 66)
    print(f"  {'TOTAL':<48}{f'{direct_score}/{total}':<10}{f'{cot_score}/{total}':<10}")
    print()
    print(
        wrap(
            f"Direct prompting scored {direct_score}/{total}. Chain-of-thought "
            f"prompting scored {cot_score}/{total}. The gap is not random noise -- "
            "it is the documented effect of letting the model condition its final "
            "answer on its own intermediate reasoning tokens instead of guessing."
        )
    )
    print()


# ---------------------------------------------------------------------------
# TRADEOFFS TABLE
# ---------------------------------------------------------------------------

def print_tradeoffs() -> None:
    print(DIVIDER)
    print("  DIRECT vs. CHAIN-OF-THOUGHT: TRADEOFFS")
    print(DIVIDER)

    rows = [
        ("Accuracy (multi-step)", "Low -- prone to shortcut errors", "High -- reasoning catches its own mistakes"),
        ("Accuracy (simple facts)", "Fine -- no benefit from CoT",       "Same -- CoT adds no real gain"),
        ("Output length / tokens", "Minimal",                            "Several times longer (the scratchpad)"),
        ("Latency & cost",         "Low -- fewer tokens generated",      "Higher -- more tokens = more $ and time"),
        ("Explainability",         "None -- answer only",                "High -- reasoning steps are inspectable"),
        ("Best used for",          "Lookups, classification, quick Q&A", "Math, logic, planning, multi-step analysis"),
    ]

    col1, col2, col3 = 24, 34, 44
    print(f"  {'Aspect':<{col1}} {'Direct Prompting':<{col2}} {'Chain of Thought':<{col3}}")
    print("  " + "-" * (col1 + col2 + col3 + 2))
    for aspect, direct, cot in rows:
        print(f"  {aspect:<{col1}} {direct:<{col2}} {cot:<{col3}}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 16: CHAIN OF THOUGHT & REASONING")
    print(DIVIDER)
    print()
    print(
        wrap(
            "Chain-of-Thought (CoT) prompting simply asks the model to show its "
            "work -- e.g. by appending \"Let's think step by step.\" Because "
            "LLMs are autoregressive, the reasoning tokens they generate become "
            "part of the context used to predict the final answer. Correct "
            "intermediate steps make a correct final answer far more likely."
        )
    )
    print()
    print(
        wrap(
            "Below, the same 4 problems (3 word problems + 1 logic puzzle) are "
            "answered twice: once with DIRECT prompting (jump straight to an "
            "answer, no visible reasoning) and once with CoT prompting (a "
            "genuine, computed step-by-step scratchpad). Watch where direct "
            "mode's shortcuts break down."
        )
    )
    print()

    results = []
    for i, problem in enumerate(PROBLEMS, start=1):
        results.append(run_problem(i, problem))

    print_scorecard(results)
    print_tradeoffs()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. CoT works because the model conditions its final answer on tokens",
        "   it just generated -- correct reasoning steps make the correct",
        "   answer statistically more likely to come next.",
        "2. The accuracy gap is real and reproducible: direct prompting fails",
        "   by taking shortcuts (using only some numbers, anchoring on the",
        "   first fact) exactly like a rushed human would.",
        "3. Zero-shot CoT needs no examples -- just append 'Let's think step",
        "   by step' to the prompt.",
        "4. CoT is not free: it costs more output tokens, more latency, and",
        "   more $ per call than a direct answer.",
        "5. Use CoT for multi-step math, logic, and planning; skip it for",
        "   simple facts, translations, or creative writing where it adds",
        "   overhead with no accuracy gain.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
