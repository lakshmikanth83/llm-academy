"""
Topic 22: Function & Tool Calling
==================================
Live demo of the tool-calling conversation loop: a calculator, a mock
weather lookup, and a mock dictionary are exposed to an assistant as
JSON-Schema tool definitions. A deterministic router simulates the model's
decision-making, three REAL Python functions execute the requested tools,
and a final message synthesizes the results — all fully offline.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import textwrap

DIVIDER = "=" * 70
THIN = "-" * 70
WRAP_W = 66


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=WRAP_W, initial_indent=indent,
                          subsequent_indent=indent)


# ---------------------------------------------------------------------------
# 1. TOOL DEFINITIONS — the JSON-Schema contract sent to the API in `tools=[...]`
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression (numbers, +, -, *, /, %, "
            "parentheses). Use this whenever the user asks for a calculation "
            "instead of guessing the answer from memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A numeric expression, e.g. '15 / 100 * 240'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Get the current weather conditions for a named city. Use this "
            "when the user asks about current weather — never guess."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo'.",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "dictionary_lookup",
        "description": (
            "Look up the definition of an English word. Use this when the "
            "user asks what a word means rather than inventing a definition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "The word to define, e.g. 'ubiquitous'.",
                }
            },
            "required": ["word"],
        },
    },
]


# ---------------------------------------------------------------------------
# 2. REAL TOOL IMPLEMENTATIONS — genuinely compute/return results
# ---------------------------------------------------------------------------

# --- Tool A: calculator — safe expression evaluation via `ast`, NOT eval() -
# Raw eval() on model-generated text is a classic security anti-pattern
# (arbitrary code execution). We parse into an AST and whitelist operators.

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_ast_node(node):
    """Recursively evaluate an AST node, rejecting anything not whitelisted."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"operator not allowed: {op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        return _ALLOWED_BINOPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError(f"operator not allowed: {op_type.__name__}")
        return _ALLOWED_UNARYOPS[op_type](_eval_ast_node(node.operand))

    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> dict:
    """
    REAL tool: safely evaluate a numeric expression. Only literals, +, -,
    *, /, %, **, unary sign, and parentheses are permitted — no names, no
    function calls, no attribute access. Never uses Python's raw eval().
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_ast_node(tree.body)
        return {"expression": expression, "result": result, "error": None}
    except ZeroDivisionError:
        return {"expression": expression, "result": None, "error": "division by zero"}
    except (SyntaxError, ValueError) as exc:
        return {"expression": expression, "result": None, "error": f"invalid expression: {exc}"}


# --- Tool B: get_weather — mock/offline data for a handful of cities -------

_MOCK_WEATHER_DB = {
    "tokyo":    {"condition": "Partly cloudy", "temp_c": 18, "humidity_pct": 62},
    "london":   {"condition": "Light rain",    "temp_c": 12, "humidity_pct": 80},
    "new york": {"condition": "Sunny",         "temp_c": 22, "humidity_pct": 45},
    "sydney":   {"condition": "Clear skies",   "temp_c": 26, "humidity_pct": 55},
    "cairo":    {"condition": "Hot and dry",   "temp_c": 34, "humidity_pct": 20},
}


def get_weather(city: str) -> dict:
    """
    REAL tool: returns canned weather data from a small in-memory table.
    Clearly MOCK/OFFLINE — a real implementation would call a weather API.
    """
    key = city.strip().lower()
    if key not in _MOCK_WEATHER_DB:
        known = ", ".join(c.title() for c in _MOCK_WEATHER_DB)
        return {"city": city, "error": f"no mock data for '{city}'. Known cities: {known}"}
    data = _MOCK_WEATHER_DB[key]
    return {"city": city.title(), "source": "MOCK/OFFLINE weather table", **data}


# --- Tool C: dictionary_lookup — mock dictionary of common words -----------

_MOCK_DICTIONARY = {
    "ubiquitous":   "present, appearing, or found everywhere.",
    "ephemeral":    "lasting for a very short time.",
    "serendipity":  "the occurrence of events by chance in a happy or beneficial way.",
    "resilient":    "able to withstand or recover quickly from difficult conditions.",
    "paradigm":     "a typical example or pattern of something; a model.",
    "cogent":       "clear, logical, and convincing.",
    "pragmatic":    "dealing with things sensibly and realistically.",
}


def dictionary_lookup(word: str) -> dict:
    """REAL tool: returns a definition from a small in-memory dictionary."""
    key = word.strip().lower().strip("'\"")
    if key not in _MOCK_DICTIONARY:
        return {"word": word, "error": f"no definition found for '{word}' in the mock dictionary."}
    return {"word": key, "definition": _MOCK_DICTIONARY[key]}


TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "dictionary_lookup": dictionary_lookup,
}


# ---------------------------------------------------------------------------
# 3. DETERMINISTIC ROUTER — stands in for the model's tool-selection step
# ---------------------------------------------------------------------------
# A real LLM decides which tools to call and with what arguments. To keep
# this demo offline and reproducible, plain pattern matching plays that role.

def _find_percentage_expression(query: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", query, re.IGNORECASE)
    if not m:
        return None, None
    pct, base = m.groups()
    return f"{pct} / 100 * {base}", f"{pct}% of {base}"


def _find_city(query: str):
    m = re.search(r"weather\s+(?:in|for)\s+([A-Za-z][A-Za-z\s]*?)(?:[\?\.,]|\s+and\b|$)", query, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _find_word(query: str):
    m = re.search(r"['‘’\"]([A-Za-z\-]+)['‘’\"]", query)
    return m.group(1) if m else None


def route_query(query: str) -> list:
    """
    Simulate the assistant's tool-selection turn: inspect the query and
    build the list of tool_use requests a real model would emit — in this
    case all three fire in the SAME turn (parallel tool calls).
    """
    requests = []

    expr, phrase = _find_percentage_expression(query)
    if expr:
        requests.append({"name": "calculator", "input": {"expression": expr}, "phrase": phrase})

    city = _find_city(query)
    if city:
        requests.append({"name": "get_weather", "input": {"city": city}, "phrase": city})

    word = _find_word(query)
    if word:
        requests.append({"name": "dictionary_lookup", "input": {"word": word}, "phrase": word})

    for i, req in enumerate(requests, start=1):
        req["type"] = "tool_use"
        req["id"] = f"toolu_{i:02d}"

    return requests


# ---------------------------------------------------------------------------
# 4. EXECUTING TOOL CALLS AND BUILDING tool_result BLOCKS
# ---------------------------------------------------------------------------

def execute_tool_call(call: dict) -> dict:
    """
    Run the real Python function behind a tool_use block and package the
    output as a tool_result block, exactly as the Anthropic API expects it
    to be sent back in the next user turn.
    """
    func = TOOL_FUNCTIONS.get(call["name"])
    if func is None:
        return {
            "type": "tool_result",
            "tool_use_id": call["id"],
            "content": json.dumps({"error": f"unknown tool '{call['name']}'"}),
            "is_error": True,
        }

    try:
        result = func(**call["input"])
        is_error = bool(result.get("error"))
    except Exception as exc:  # tool crashed — never let this take down the loop
        result = {"error": f"tool raised an exception: {exc}"}
        is_error = True

    return {
        "type": "tool_result",
        "tool_use_id": call["id"],
        "content": json.dumps(result),
        "is_error": is_error,
    }


def synthesize_final_answer(calls: list, results_by_id: dict) -> str:
    """
    Simulate the model's final turn: weave every tool_result into one
    coherent, grounded answer. A real model would generate this text
    itself; here we template it from the actual (real) tool outputs.
    """
    sentences = []
    for call in calls:
        res = json.loads(results_by_id[call["id"]])

        if call["name"] == "calculator":
            if res.get("error"):
                sentences.append(f"I couldn't compute {call['phrase']}: {res['error']}.")
            else:
                value = res["result"]
                value_str = f"{value:g}" if isinstance(value, float) else str(value)
                sentences.append(f"{call['phrase']} is {value_str}.")

        elif call["name"] == "get_weather":
            if res.get("error"):
                sentences.append(f"I couldn't get the weather for {call['phrase']}: {res['error']}")
            else:
                sentences.append(
                    f"The weather in {res['city']} is currently {res['condition'].lower()}, "
                    f"{res['temp_c']}°C with {res['humidity_pct']}% humidity."
                )

        elif call["name"] == "dictionary_lookup":
            if res.get("error"):
                sentences.append(res["error"])
            else:
                sentences.append(f"'{res['word']}' means: {res['definition']}")

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# 5. DEMO — the full multi-tool conversation loop, turn by turn
# ---------------------------------------------------------------------------

def print_tool_definitions() -> None:
    print("  Tool definitions sent to the API in `tools=[...]`:")
    print(THIN)
    for tool in TOOL_DEFINITIONS:
        print(json.dumps(tool, indent=2))
        print()


def run_conversation(query: str) -> None:
    print(DIVIDER)
    print(f"  USER QUERY: {query}")
    print(DIVIDER)

    # ---- Turn 1: user message ---------------------------------------------
    print()
    print("  TURN 1 — user message")
    print(THIN)
    print(f'  {{"role": "user", "content": "{query}"}}')

    # ---- Turn 2: assistant emits parallel tool_use blocks ------------------
    print()
    print("  TURN 2 — assistant message (stop_reason = 'tool_use')")
    print(THIN)
    tool_calls = route_query(query)
    print("  The model decided it needs ALL THREE tools in a single turn")
    print("  (parallel tool calls — one round trip instead of three):")
    print()
    for call in tool_calls:
        block = {"type": call["type"], "id": call["id"], "name": call["name"], "input": call["input"]}
        print(json.dumps(block, indent=2))
        print()

    # ---- Executing the real tools ------------------------------------------
    print("  EXECUTING TOOLS (real Python functions — not simulated)")
    print(THIN)
    results_by_id = {}
    tool_result_blocks = []
    for call in tool_calls:
        result_block = execute_tool_call(call)
        results_by_id[call["id"]] = result_block["content"]
        tool_result_blocks.append(result_block)
        status = "ERROR" if result_block["is_error"] else "ok"
        print(f"  {call['name']}(**{call['input']})  ->  [{status}] {result_block['content']}")
    print()

    # ---- Turn 3: user message carrying all tool_result blocks --------------
    print("  TURN 3 — user message (tool_result blocks, sent together)")
    print(THIN)
    print(json.dumps(tool_result_blocks, indent=2))
    print()

    # ---- Turn 4: final assistant message ------------------------------------
    print("  TURN 4 — final assistant message (stop_reason = 'end_turn')")
    print(THIN)
    final_answer = synthesize_final_answer(tool_calls, results_by_id)
    print(wrap(final_answer))
    print()


def demo_round_trip_comparison() -> None:
    print(DIVIDER)
    print("  PARALLEL vs SEQUENTIAL TOOL CALLS")
    print(DIVIDER)
    print()
    rows = [
        ("Tools needed",        "3 (calculator, weather, dictionary)", "3 (same tools)"),
        ("Tool calls per turn", "All 3 in ONE assistant message",      "1 per assistant message"),
        ("API round trips",     "2  (tools -> results -> answer)",     "4  (call, call, call, answer)"),
        ("Latency",             "Lower — tools run concurrently",      "Higher — each waits on the last"),
        ("When to use",         "Independent tool calls (default)",    "Tool B needs tool A's output"),
    ]
    col1, col2, col3 = 20, 38, 34
    print(f"  {'Aspect':<{col1}} {'Parallel (this demo)':<{col2}} {'Sequential':<{col3}}")
    print("  " + "-" * (col1 + col2 + col3))
    for aspect, par, seq in rows:
        print(f"  {aspect:<{col1}} {par:<{col2}} {seq:<{col3}}")
    print(
        "\n  Independent tool calls belong in one turn; sequential round trips\n"
        "  are only needed when one tool's output feeds the next tool's input."
    )
    print()


# ---------------------------------------------------------------------------
# 6. WHY IT MATTERS + COMMON PITFALLS
# ---------------------------------------------------------------------------

def print_why_it_matters() -> None:
    print(DIVIDER)
    print("  WHY TOOL CALLING MATTERS")
    print(DIVIDER)
    print(
        wrap(
            "Without tools, an LLM can only answer from what it memorized during "
            "training. Ask it '15% of 240' and it might get lucky; ask it today's "
            "weather in Tokyo or a niche word's definition and it can only guess — "
            "producing a fluent but potentially wrong (hallucinated) answer. Tool "
            "calling grounds every part of the response in a real computation or a "
            "real data source: the calculator can't be wrong about arithmetic, the "
            "weather tool returns actual (here, mock) current data, and the "
            "dictionary returns an actual definition instead of an invented one."
        )
    )
    print()


def print_pitfalls() -> None:
    print(DIVIDER)
    print("  COMMON PITFALLS")
    print(DIVIDER)
    pitfalls = [
        "1. Never eval() model-generated strings — treat tool_use input as",
        "   untrusted, exactly like user input. This demo's calculator parses",
        "   an AST and whitelists operators instead of calling eval().",
        "2. Validate every argument's type and range before running a tool —",
        "   a malformed or adversarial 'city' or 'expression' should fail",
        "   safely, not crash the process or leak internal errors.",
        "3. Handle tool errors gracefully: return an is_error tool_result",
        "   instead of raising, so the model can recover or apologize rather",
        "   than the whole conversation loop crashing.",
        "4. Whoever's code executes the tool_use request owns the security",
        "   boundary — the model only ever *asks*; your application decides",
        "   what is actually allowed to run.",
        "5. Always send tool_result blocks with a matching tool_use_id, and",
        "   send all of them together when the calls were parallel.",
    ]
    for line in pitfalls:
        print(f"  {line}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

DEMO_QUERY = "What's 15% of 240? Also what's the weather in Tokyo, and what does 'ubiquitous' mean?"


def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 22: FUNCTION & TOOL CALLING — OFFLINE DEMO")
    print(DIVIDER)
    print()
    print(
        wrap(
            "This demo simulates the tool-calling LOOP mechanics used by real LLM "
            "APIs, without calling any API. Three genuinely working Python tools "
            "(calculator, mock weather, mock dictionary) are described to an "
            "assistant via JSON-Schema tool definitions. A deterministic router "
            "stands in for the model's decision-making, then the real conversation "
            "loop — tool_use, tool execution, tool_result, final answer — plays out "
            "turn by turn."
        )
    )
    print()

    print_tool_definitions()
    run_conversation(DEMO_QUERY)
    demo_round_trip_comparison()
    print_why_it_matters()
    print_pitfalls()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. A tool definition is name + description + input_schema — the",
        "   description is what actually drives the model's tool choice.",
        "2. tool_use blocks are requests, not executions — YOUR code runs",
        "   the tool and reports back via a tool_result block.",
        "3. Independent tools can be called in parallel within one turn,",
        "   cutting round trips versus calling them one at a time.",
        "4. Never eval() model output — parse and whitelist, as this demo's",
        "   AST-based calculator does.",
        "5. Grounding answers in real tool output (math, data lookups) is",
        "   what turns a plausible-sounding chatbot into a trustworthy agent.",
        "6. Every tool call should handle failure gracefully via is_error",
        "   tool_result blocks instead of raising exceptions upward.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
