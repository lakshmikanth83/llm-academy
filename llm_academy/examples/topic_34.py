"""
Topic 34: ReAct & Planning
==========================
A from-scratch ReAct (Reasoning + Acting) agent loop: Thought -> Action ->
Observation, repeated until the agent has gathered enough information to
answer. Includes a deliberate self-correction step triggered by an
ambiguous search result, plus a comparison of ReAct vs. pure Chain-of-
Thought vs. Plan-and-Execute.
"""

from __future__ import annotations

import ast
import operator
import re
import textwrap

DIVIDER = "=" * 70
THIN = "-" * 70


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=66, initial_indent=indent,
                         subsequent_indent=indent)


# ---------------------------------------------------------------------------
# TOOL 1: search() — a tiny "search engine" over a hardcoded fact database
# ---------------------------------------------------------------------------
# Each entry has a set of keywords and a canned text snippet. Retrieval uses
# Jaccard word-overlap similarity (same idea as a real search index, just
# without embeddings). Crucially, ONE entry ("population of Paris") is
# deliberately ambiguous — it does not contain a specific number — so the
# agent must recognise the gap and issue a follow-up query.

_STOPWORDS = {"a", "an", "the", "is", "of", "in", "on", "to", "and", "for"}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


FACTS = [
    {
        "keywords": {"capital", "france"},
        "text": "France's capital city is Paris, often nicknamed 'The City of Light'.",
    },
    {
        # Deliberately ambiguous — no concrete number, forces self-correction.
        "keywords": {"population", "paris"},
        "text": (
            "Population figures for Paris vary widely depending on the "
            "definition used: the city proper, the urban unit, and the "
            "wider metro area all report very different numbers. Please "
            "specify which definition you mean."
        ),
    },
    {
        "keywords": {"population", "paris", "city", "proper"},
        "text": (
            "The city of Paris, within its official city-limit boundaries, "
            "has a population of approximately 2,148,000 people (2023 estimate)."
        ),
    },
    {
        "keywords": {"capital", "portugal"},
        "text": "Portugal's capital city is Lisbon.",
    },
    {
        "keywords": {"population", "lisbon"},
        "text": (
            "Lisbon has a population of approximately 545,000 people "
            "within city limits (2023 estimate)."
        ),
    },
]


def search(query: str) -> str:
    """Return the single best-matching fact snippet for a query."""
    q_tokens = _tokenize(query)
    scored = [(_jaccard(q_tokens, fact["keywords"]), fact) for fact in FACTS]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_fact = scored[0]
    if best_score == 0.0:
        return "No relevant information found for that query."
    return best_fact["text"]


# ---------------------------------------------------------------------------
# TOOL 2: calculate() — a SAFE arithmetic evaluator built on the ast module
# ---------------------------------------------------------------------------
# Never use raw eval() on untrusted / model-generated strings. Instead, parse
# the expression into an AST and only allow a whitelist of numeric operators.

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        return _ALLOWED_BINOPS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
        return _ALLOWED_UNARYOPS[op_type](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Safely evaluate a numeric expression (+ - * / ** only) and return a string result."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"Error evaluating '{expression}': {exc}"
    if isinstance(result, float):
        result = round(result, 4)
    return f"{expression} = {result}"


# ---------------------------------------------------------------------------
# FACT EXTRACTION — parse the agent's own observations to update its state
# ---------------------------------------------------------------------------
# The brain never "knows" the answer up front. It re-derives what it has
# learned so far purely by pattern-matching the text the tools returned.

_FRANCE_CAPITAL_RE = re.compile(r"France's capital city is (\w+)")
_PORTUGAL_CAPITAL_RE = re.compile(r"Portugal's capital city is (\w+)")
_PARIS_POP_RE = re.compile(r"Paris.*?population of approximately ([\d,]+)")
_LISBON_POP_RE = re.compile(r"Lisbon.*?population of approximately ([\d,]+)")
_CALC_RESULT_RE = re.compile(r"=\s*(-?[\d.]+)\s*$")


def _extract_facts(history: list[dict]) -> dict:
    """Scan every past Observation and pull out whatever facts we can find."""
    facts: dict = {}
    for step in history:
        obs = step["observation"] or ""
        action_name, action_arg = step["action"]

        m = _FRANCE_CAPITAL_RE.search(obs)
        if m:
            facts["capital_france"] = m.group(1)

        m = _PORTUGAL_CAPITAL_RE.search(obs)
        if m:
            facts["capital_portugal"] = m.group(1)

        m = _PARIS_POP_RE.search(obs)
        if m:
            facts["pop_paris"] = int(m.group(1).replace(",", ""))

        m = _LISBON_POP_RE.search(obs)
        if m:
            facts["pop_lisbon"] = int(m.group(1).replace(",", ""))

        if action_name == "calculate":
            m = _CALC_RESULT_RE.search(obs)
            if m:
                facts["ratio"] = float(m.group(1))

    return facts


# ---------------------------------------------------------------------------
# AGENT BRAIN — decides the NEXT Thought + Action from the accumulated history
# ---------------------------------------------------------------------------
# This is a deterministic decision tree (no real LLM is available offline),
# but it genuinely branches on the content of past Observations rather than
# replaying a fixed script: which fact is still missing, and whether the
# most recent search came back ambiguous, both change what happens next.

def decide_next(history: list[dict]):
    """
    Returns (thought, action, final_answer):
      - action is (tool_name, tool_arg) when more work is needed, final_answer is None
      - action is None and final_answer is a string when the agent is done
    """
    facts = _extract_facts(history)

    paris_pop_search_attempts = sum(
        1
        for step in history
        if step["action"][0] == "search"
        and "population" in step["action"][1].lower()
        and "paris" in step["action"][1].lower()
    )

    if "capital_france" not in facts:
        thought = (
            "The task asks about the capital of France. I don't know it yet, "
            "so my first step should be to search for it."
        )
        return thought, ("search", "capital of France"), None

    if "pop_paris" not in facts:
        if paris_pop_search_attempts == 0:
            thought = (
                f"I know the capital of France is {facts['capital_france']}. "
                f"Next I need its population."
            )
            return thought, ("search", f"population of {facts['capital_france']}"), None
        else:
            thought = (
                "That result was ambiguous — it listed several different "
                "definitions of Paris instead of a single number. I should "
                "refine my query and ask specifically for the population "
                "within official city limits."
            )
            return thought, ("search", f"population of {facts['capital_france']} city proper"), None

    if "capital_portugal" not in facts:
        thought = "Now I need the capital of Portugal to find its population."
        return thought, ("search", "capital of Portugal"), None

    if "pop_lisbon" not in facts:
        thought = (
            f"I know the capital of Portugal is {facts['capital_portugal']}. "
            f"Next I need its population."
        )
        return thought, ("search", f"population of {facts['capital_portugal']}"), None

    if "ratio" not in facts:
        thought = (
            f"I now have both populations: {facts['capital_france']} "
            f"~{facts['pop_paris']:,} and {facts['capital_portugal']} "
            f"~{facts['pop_lisbon']:,}. To answer the 'more than double' "
            f"part of the question, I need their ratio."
        )
        return thought, ("calculate", f"{facts['pop_paris']} / {facts['pop_lisbon']}"), None

    ratio = facts["ratio"]
    more_than_double = ratio > 2
    thought = "I now have enough information to answer the question."
    answer = (
        f"The capital of France is {facts['capital_france']} "
        f"(population ~{facts['pop_paris']:,}), and the capital of Portugal "
        f"is {facts['capital_portugal']} (population ~{facts['pop_lisbon']:,}). "
        f"{facts['capital_france']}'s population is about {ratio:.2f}x "
        f"{facts['capital_portugal']}'s — that is "
        f"{'MORE' if more_than_double else 'NOT more'} than double."
    )
    return thought, None, answer


# ---------------------------------------------------------------------------
# REACT LOOP RUNNER
# ---------------------------------------------------------------------------

def run_react_loop(task: str, max_iterations: int = 8) -> list[dict]:
    print(THIN)
    print(f"  TASK: {task}")
    print(THIN)
    print()

    history: list[dict] = []

    for i in range(1, max_iterations + 1):
        thought, action, final_answer = decide_next(history)

        print(f"  Iteration {i}")
        print(wrap(f"Thought: {thought}"))

        if action is None:
            print(wrap(f"Final Answer: {final_answer}"))
            print()
            break

        tool_name, tool_arg = action
        print(f"  Action: {tool_name}({tool_arg!r})")

        if tool_name == "search":
            observation = search(tool_arg)
        elif tool_name == "calculate":
            observation = calculate(tool_arg)
        else:
            observation = f"Unknown tool '{tool_name}'."

        print(wrap(f"Observation: {observation}"))
        print()

        history.append({"thought": thought, "action": action, "observation": observation})
    else:
        print("  Max iterations reached without a final answer.")

    return history


# ---------------------------------------------------------------------------
# COMPARISON: ReAct vs. pure Chain-of-Thought vs. Plan-and-Execute
# ---------------------------------------------------------------------------

def demo_comparison() -> None:
    print(DIVIDER)
    print("  WHY REACT WORKS — AND WHEN TO USE SOMETHING ELSE")
    print(DIVIDER)
    print()
    print(wrap(
        "ReAct's core value is interleaving reasoning and acting: after every "
        "tool call, the agent re-reads what actually happened before deciding "
        "what to do next. In the trace above, iteration 2's search came back "
        "ambiguous — a pure 'plan everything up front, then execute blindly' "
        "agent would have had no mechanism to notice that and would have "
        "either crashed on a missing number or hallucinated one. Because "
        "ReAct grounds each Thought in a real Observation, it caught the gap "
        "and corrected course on the very next step."
    ))
    print()

    rows = [
        ("Reasons before acting", "Yes", "Yes", "Yes (once, up front)"),
        ("Takes real actions", "No", "Yes", "Yes"),
        ("Sees results before next step", "N/A", "Yes", "No (until re-plan)"),
        ("Self-corrects mid-task", "No", "Yes", "Only if a replanner runs"),
        ("Token cost per task", "Low", "Higher (many round-trips)", "Medium"),
        ("Best for", "Pure reasoning/math", "Multi-hop lookups, unknown unknowns", "Well-understood, stable workflows"),
    ]
    col1, col2, col3, col4 = 32, 21, 38, 35
    print(f"  {'Property':<{col1}} {'CoT':<{col2}} {'ReAct':<{col3}} {'Plan-and-Execute':<{col4}}")
    print("  " + "-" * (col1 + col2 + col3 + col4))
    for prop, cot, react, plan in rows:
        print(f"  {prop:<{col1}} {cot:<{col2}} {react:<{col3}} {plan:<{col4}}")
    print()

    print(wrap(
        "Rule of thumb: use pure Chain-of-Thought when no external tools or "
        "facts are needed (e.g. arithmetic, logic puzzles). Use ReAct when "
        "the task requires multiple dependent lookups and the outcome of one "
        "step can change what you do next. Use Plan-and-Execute when the "
        "steps are well understood in advance and you want to minimize "
        "back-and-forth token cost — optionally adding a replanner for when "
        "reality still deviates from the plan."
    ))
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 34: REACT & PLANNING — LIVE THOUGHT/ACTION/OBSERVATION TRACE")
    print(DIVIDER)
    print()
    print(wrap(
        "ReAct (Reasoning + Acting) interleaves a Thought, an Action (tool "
        "call), and an Observation (tool result) in a repeating loop. Each "
        "Thought is generated AFTER seeing the previous Observation, which "
        "is what lets the agent notice mistakes and self-correct instead of "
        "following a rigid, pre-written script."
    ))
    print()
    print("  Tools available to the agent:")
    print("    - search(query)     -> looks up a snippet in a small fact database")
    print("    - calculate(expr)   -> evaluates arithmetic safely via the ast module")
    print()

    task = (
        "What is the population of the capital of France, and is it more "
        "than double the population of the capital of Portugal?"
    )

    history = run_react_loop(task)

    print(THIN)
    print(wrap(
        "Notice iteration 2 above: searching 'population of Paris' returned "
        "an ambiguous snippet with no concrete number. The agent's next "
        "Thought explicitly named that gap and issued a refined search "
        "('population of Paris city proper') rather than guessing — this is "
        "the self-correction ReAct is designed to enable."
    ))
    print(THIN)
    print()
    print(f"  Total iterations executed: {len(history) + 1}")
    print()

    demo_comparison()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. ReAct repeats Thought -> Action -> Observation, and each new",
        "   Thought is conditioned on the real result of the previous Action.",
        "2. This makes self-correction possible: an ambiguous or wrong",
        "   Observation can trigger a refined follow-up action instead of a",
        "   hallucinated answer.",
        "3. Never eval() untrusted expressions — parse with ast and whitelist",
        "   the exact operators you intend to support, as calculate() does.",
        "4. Pure Chain-of-Thought reasons but cannot act; Plan-and-Execute",
        "   acts on a fixed plan but reacts slowly to surprises; ReAct",
        "   reasons and acts on every single step.",
        "5. Making Thoughts explicit (rather than hidden) is what makes an",
        "   agent's behavior interpretable and debuggable by a human.",
        "6. This entire agent — tools, brain, and loop — used zero external",
        "   dependencies and zero real network calls.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
