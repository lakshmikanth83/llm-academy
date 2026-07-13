"""
Topic 18: System Prompts & Roles
=================================
Simulates a "persona engine" that runs the SAME user questions through three
different system prompts and shows how the persona/constraints change the
response — without ever calling a real LLM. Also verifies programmatically
whether each persona actually obeys the constraints it was given.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Callable

DIVIDER = "=" * 70
THIN = "-" * 70
WRAP_W = 66


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=WRAP_W, initial_indent=indent,
                         subsequent_indent=indent)


# ---------------------------------------------------------------------------
# 1. CONTENT FACTS — persona-neutral knowledge, keyed by question topic.
#    Personas do NOT get their own hardcoded full answers; every persona
#    renders its response FROM these same facts, styled differently.
# ---------------------------------------------------------------------------

CONTENT_FACTS = {
    "password_reset": {
        # Full clauses (no trailing punctuation) used by the verbose personas.
        "core": [
            "to reset your password, go to the login page and click the "
            "'forgot password' link",
            "you will then receive an email with a secure reset link, so "
            "open it and choose a new password",
            "if the email does not arrive within a few minutes, check your "
            "spam folder before requesting another one",
        ],
        # Short clauses used by the terse, constrained persona.
        "terse": [
            "click 'forgot password' on the login page",
            "then follow the emailed link to set a new password",
        ],
    },
    "math_percent": {
        "core": [
            "to find 15% of 80, multiply 80 by 0.15",
            "0.15 times 80 equals 12, so 15% of 80 is 12",
        ],
        "terse": [
            "15% of 80 is 12",
            "0.15 x 80 = 12",
        ],
    },
    "code_review": {
        "core": [
            "the function correctly returns the sum of its two arguments "
            "for numeric input",
            "it has no type hints, no docstring, and no input validation, "
            "so a non-numeric argument will raise a runtime type error",
            "consider adding type hints such as (a: float, b: float) -> "
            "float and a short docstring for clarity",
        ],
        "terse": [
            "logic is correct for numeric input",
            "add type hints and a docstring; non-numeric args will raise a type error",
        ],
    },
}

QUESTIONS = [
    {"topic": "password_reset", "text": "How do I reset my password?"},
    {"topic": "math_percent", "text": "What's 15% of 80?"},
    {"topic": "code_review",
     "text": "Can you review this code: `def add(a, b): return a+b`"},
]


# ---------------------------------------------------------------------------
# 2. STYLE HELPERS — small, reusable building blocks. Personas are built by
#    composing these with CONTENT_FACTS, not by pasting finished answers.
# ---------------------------------------------------------------------------

def to_sentence(clause: str) -> str:
    """Capitalize the first letter and terminate with a period."""
    clause = clause.strip()
    if not clause:
        return ""
    return clause[0].upper() + clause[1:].rstrip(".") + "."


_PIRATE_SUBSTITUTIONS = [
    (r"\byou will\b", "ye will"),
    (r"\byou\b", "ye"),
    (r"\byour\b", "yer"),
    (r"\bmy\b", "me"),
]


def piratize(clause: str) -> str:
    """Apply pirate-vocabulary word substitutions to a plain-English clause."""
    result = clause
    for pattern, replacement in _PIRATE_SUBSTITUTIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def count_sentences(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([p for p in parts if p.strip()])


_PLEASANTRY_PHRASES = [
    "happy to help", "let me know", "sure,", "of course",
    "glad to", "no problem", "great question",
]


def has_pleasantry(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _PLEASANTRY_PHRASES)


_PIRATE_MARKERS = ["arr", "matey", " ye ", " ye'", "yer ", "ye will"]


def has_pirate_vocab(text: str) -> bool:
    lower = " " + text.lower() + " "
    return any(m in lower for m in _PIRATE_MARKERS)


# ---------------------------------------------------------------------------
# 3. RESPONSE RENDERERS — one per persona, keyed off topic. Each builds a
#    response deterministically from CONTENT_FACTS[topic] + persona style.
# ---------------------------------------------------------------------------

def render_neutral(topic: str) -> str:
    """No system prompt: a generic, mildly chatty default assistant voice."""
    clauses = CONTENT_FACTS[topic]["core"]
    body = " ".join(to_sentence(c) for c in clauses)
    return f"Sure, happy to help! {body} Let me know if you need anything else!"


def render_pirate(topic: str) -> str:
    """Friendly pirate customer-support persona: same facts, pirate voice."""
    clauses = CONTENT_FACTS[topic]["core"]
    body = " ".join(to_sentence(piratize(c)) for c in clauses)
    return f"Arrr, matey! {body} Fair winds to ye!"


def render_engineer(topic: str) -> str:
    """Terse senior engineer persona: max 2 sentences, zero pleasantries."""
    clauses = CONTENT_FACTS[topic]["terse"]
    sentences = [to_sentence(c) for c in clauses[:2]]
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# 4. PERSONAS — the "system prompt" is the actual instruction text a real
#    API call would send in the {"role": "system", ...} message.
# ---------------------------------------------------------------------------

@dataclass
class Persona:
    key: str
    label: str
    system_prompt: str
    render: Callable[[str], str]
    constraint_summary: str


def check_constraints(persona_key: str, text: str) -> tuple[str, str]:
    """Programmatically verify whether a rendered response obeys its
    persona's system-prompt constraints. Returns (verdict, detail)."""
    if persona_key == "none":
        return "PASS", "No system prompt was set — nothing to verify."

    if persona_key == "pirate":
        ok = has_pirate_vocab(text)
        return ("PASS" if ok else "FAIL",
                f"Requires pirate vocabulary (e.g. 'ye', 'matey', 'arrr') — found: {ok}")

    if persona_key == "engineer":
        n = count_sentences(text)
        pleasant = has_pleasantry(text)
        ok = (n <= 2) and (not pleasant)
        detail = (f"Requires <=2 sentences and no pleasantries — "
                  f"sentence count: {n}, pleasantry found: {pleasant}")
        return ("PASS" if ok else "FAIL", detail)

    return "PASS", ""


PERSONAS = [
    Persona(
        key="none",
        label="No System Prompt (Neutral Assistant)",
        system_prompt="(none set — default, unconstrained assistant behavior)",
        render=render_neutral,
        constraint_summary="No constraints to enforce.",
    ),
    Persona(
        key="pirate",
        label="Friendly Pirate Support Agent",
        system_prompt=(
            "You are Salty Sam, a friendly customer-support agent who talks "
            "like a pirate. Use pirate vocabulary ('ye', 'yer', 'arrr', "
            "'matey') in every response. Stay warm, patient, and helpful, "
            "and sign off with a nautical farewell."
        ),
        render=render_pirate,
        constraint_summary="Must use pirate vocabulary throughout.",
    ),
    Persona(
        key="engineer",
        label="Terse Senior Engineer (Code Reviewer)",
        system_prompt=(
            "You are a terse senior software engineer performing a code "
            "review. Answer in at most 2 sentences. No greetings, no "
            "pleasantries, no filler — state the technical point directly."
        ),
        render=render_engineer,
        constraint_summary="Must use <=2 sentences and no pleasantries.",
    ),
]


# ---------------------------------------------------------------------------
# 5. DEMO — system prompt intro
# ---------------------------------------------------------------------------

def demo_system_prompts() -> None:
    print(DIVIDER)
    print("  THE THREE SYSTEM PROMPTS BEING COMPARED")
    print(DIVIDER)
    for p in PERSONAS:
        print()
        print(f"  [{p.key}] {p.label}")
        print(THIN)
        print(wrap(p.system_prompt))
        print(f"  Constraint to verify: {p.constraint_summary}")
    print()


# ---------------------------------------------------------------------------
# 6. DEMO — run every question through every persona, side by side
# ---------------------------------------------------------------------------

def demo_question(question: dict, num: int) -> list[tuple[str, str, str, str]]:
    """Render + verify all personas for one question. Returns rows of
    (persona_label, response, verdict, detail) for the final summary."""
    print(DIVIDER)
    print(f"  QUESTION {num}: {question['text']}")
    print(DIVIDER)

    rows = []
    for p in PERSONAS:
        response = p.render(question["topic"])
        verdict, detail = check_constraints(p.key, response)

        print()
        print(f"  PERSONA: {p.label}")
        print(THIN)
        print(wrap(response))
        print(f"  Constraint check: [{verdict}]  {detail}")

        rows.append((p.label, response, verdict, detail))

    print()
    print("  SIDE-BY-SIDE SUMMARY")
    print(THIN)
    col1, col2, col3 = 40, 11, 7
    print(f"  {'Persona':<{col1}} {'Sentences':<{col2}} {'Verdict':<{col3}}")
    print("  " + "-" * (col1 + col2 + col3))
    for label, response, verdict, _ in rows:
        n = count_sentences(response)
        print(f"  {label:<{col1}} {n:<{col2}} {verdict:<{col3}}")
    print()

    return [(question["text"], label, verdict, detail) for label, _, verdict, detail in rows]


# ---------------------------------------------------------------------------
# 7. DEMO — aggregate constraint-adherence results across all questions
# ---------------------------------------------------------------------------

def demo_constraint_summary(all_rows: list[tuple[str, str, str, str]]) -> None:
    print(DIVIDER)
    print("  CONSTRAINT-FOLLOWING CHECK — ACROSS ALL QUESTIONS")
    print(DIVIDER)
    print(
        "  Does each persona actually obey the rules stated in its system\n"
        "  prompt, or does it just 'sound' like the persona? Verified by\n"
        "  code, not by eyeballing the text.\n"
    )

    col1, col2, col3 = 40, 30, 8
    print(f"  {'Persona':<{col1}} {'Question':<{col2}} {'Verdict':<{col3}}")
    print("  " + "-" * (col1 + col2 + col3))
    fail_count = 0
    for question_text, label, verdict, _ in all_rows:
        short_q = (question_text[:col2 - 1] + "…") if len(question_text) > col2 else question_text
        print(f"  {label:<{col1}} {short_q:<{col2}} {verdict:<{col3}}")
        if verdict == "FAIL":
            fail_count += 1

    print()
    total = len(all_rows)
    print(f"  Result: {total - fail_count}/{total} responses PASSED their persona's constraints.")
    print(
        "\n  Note: the neutral persona has no constraints by definition, so it\n"
        "  always PASSes trivially — the interesting checks are the pirate\n"
        "  persona's vocabulary rule and the engineer persona's 2-sentence,\n"
        "  no-pleasantries rule."
    )
    print()


# ---------------------------------------------------------------------------
# 8. COMPARISON — system prompt vs. user-message instructions
# ---------------------------------------------------------------------------

def demo_when_to_use() -> None:
    print(DIVIDER)
    print("  WHEN TO USE A SYSTEM PROMPT VS. A USER-MESSAGE INSTRUCTION")
    print(DIVIDER)

    rows = [
        ("Scope",           "Whole conversation, every turn",   "Just this one turn"),
        ("Who sets it",     "Application developer",            "End user (or injected per call)"),
        ("Authority",       "Higher — models trained to obey",   "Lower — bounded by system rules"),
        ("Visibility",      "Usually hidden from the user",      "Always visible in the transcript"),
        ("Repetition cost", "Resent on every API call",          "Sent once, only when needed"),
        ("Best for",        "Persona, scope, hard constraints",  "One-off tasks, ad-hoc formatting"),
    ]

    col1, col2, col3 = 16, 34, 34
    print(f"  {'Aspect':<{col1}} {'System Prompt':<{col2}} {'User Message':<{col3}}")
    print("  " + "-" * (col1 + col2 + col3))
    for aspect, sysp, usrp in rows:
        print(f"  {aspect:<{col1}} {sysp:<{col2}} {usrp:<{col3}}")

    print(
        "\n  Tradeoffs:\n"
        "    - System prompts are the right place for anything that must\n"
        "      hold across every turn (persona, scope limits, output format).\n"
        "    - Per-turn, situational instructions (\"summarize this in 3\n"
        "      bullets\") usually belong in the user message — repeating them\n"
        "      in the system prompt would be wasted, static overhead.\n"
        "    - System prompts carry more authority but are not a hard\n"
        "      guarantee: genuinely non-negotiable safety rules still need\n"
        "      model-level training, not just instructions, plus adversarial\n"
        "      testing against prompt-injection attempts.\n"
        "    - A bloated system prompt dilutes attention on the instructions\n"
        "      that matter most — keep it focused."
    )
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 18: SYSTEM PROMPTS & ROLES")
    print(DIVIDER)
    print()
    print(
        "  A system prompt is invisible instruction text sent once, at the\n"
        "  start of a conversation, that shapes every response the model\n"
        "  gives afterward — persona, tone, vocabulary, and hard rules.\n"
        "  This demo simulates 3 personas (no real API calls) and runs the\n"
        "  SAME 3 user questions through each one, then verifies by code\n"
        "  whether each persona actually followed its own constraints."
    )
    print()

    demo_system_prompts()

    all_rows: list[tuple[str, str, str, str]] = []
    for i, question in enumerate(QUESTIONS, start=1):
        all_rows.extend(demo_question(question, i))

    demo_constraint_summary(all_rows)
    demo_when_to_use()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. The system prompt is invisible to the user but shapes every reply —",
        "   the SAME underlying facts came out as 3 completely different voices.",
        "2. System instructions carry more authority than user messages in",
        "   well-trained models, but that authority should be tested, not assumed.",
        "3. Constraints stated in a system prompt (tone, length, vocabulary) are",
        "   only real if you verify them programmatically — 'sounding right' is",
        "   not the same as passing a check, as the PASS/FAIL table showed.",
        "4. Persona, scope, and hard rules belong in the system prompt because",
        "   they must persist across every turn without being resent by the user.",
        "5. One-off, per-turn requests are usually cheaper and clearer as user",
        "   messages rather than bloating a persistent system prompt.",
        "6. Hard constraints ('never reveal X') need adversarial testing —",
        "   a system prompt is an instruction, not an unbreakable guarantee.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
