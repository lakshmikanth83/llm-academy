"""
Topic 47: Guardrails
====================
Demonstrates input and output guardrails for LLM applications.

Input guardrails:
  - BLOCK  — explicit harmful keywords, PII patterns (SSN, credit card, email)
  - WARN   — vague or ambiguous harmful intent

Output guardrails:
  - Scan the model's response for sensitive information before returning it

All LLM responses are simulated so no API key is required.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------

class Verdict(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class GuardrailResult:
    verdict: Verdict
    reason:  str
    detail:  str = ""


# ---------------------------------------------------------------------------
# INPUT GUARDRAIL
# ---------------------------------------------------------------------------

# --- Block patterns --------------------------------------------------------
_HARMFUL_KEYWORDS = [
    r"\b(how\s+to\s+make\s+(a\s+)?(bomb|explosive|poison|weapon))\b",
    r"\b(kill|murder|assassinate)\s+\w+",
    r"\b(synthesize|manufacture)\s+(meth|fentanyl|cocaine|heroin|drugs?)\b",
    r"\b(hack|exploit)\s+(a\s+)?(bank|hospital|power\s+grid|water\s+supply)\b",
    r"\b(child\s+abuse|csam|child\s+pornography)\b",
    r"\bsuicide\s+(method|how|guide|instruction)\b",
]

_PII_PATTERNS = {
    "SSN":         r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "Credit card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "Email":       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "Phone (US)":  r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "IP address":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

# --- Warn patterns ---------------------------------------------------------
_WARN_PATTERNS = [
    (r"\b(hurt|harm|attack)\s+\w+", "Potentially harmful intent toward a person"),
    (r"\b(bypass|circumvent|disable)\s+(security|safety|filter|guardrail)\b",
     "Attempting to bypass safety systems"),
    (r"\b(illegal|unlawful)\s+(way|method|trick)\b",
     "Request may involve illegal activity"),
    (r"\bwithout\s+(getting\s+)?(caught|detected|noticed)\b",
     "Evasion-oriented language"),
    (r"\b(trick|deceive|manipulate)\s+(someone|a person|people)\b",
     "Potential social-engineering intent"),
]


def check_input(text: str) -> GuardrailResult:
    """Run all input checks; return the most severe verdict found."""
    lower = text.lower()

    # 1. Harmful keywords — FAIL immediately
    for pattern in _HARMFUL_KEYWORDS:
        m = re.search(pattern, lower)
        if m:
            return GuardrailResult(
                verdict=Verdict.FAIL,
                reason="Harmful content detected",
                detail=f"Matched pattern: '{m.group(0)}'",
            )

    # 2. PII — FAIL immediately
    for pii_name, pattern in _PII_PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            return GuardrailResult(
                verdict=Verdict.FAIL,
                reason=f"PII detected: {pii_name}",
                detail=f"Found: '{m.group(0)[:20]}...' — do not send personal data to an LLM",
            )

    # 3. Vague harmful intent — WARN
    for pattern, description in _WARN_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            return GuardrailResult(
                verdict=Verdict.WARN,
                reason="Possible harmful intent",
                detail=f"{description}  (matched: '{m.group(0)}')",
            )

    return GuardrailResult(verdict=Verdict.PASS, reason="No issues detected")


# ---------------------------------------------------------------------------
# SIMULATED LLM RESPONSES
# ---------------------------------------------------------------------------

def simulate_llm(user_input: str) -> str:
    """Return a plausible but harmless simulated response."""
    lower = user_input.lower()
    if "weather" in lower:
        return "The weather in San Francisco today is 65°F and partly cloudy."
    if "capital" in lower and "france" in lower:
        return "The capital of France is Paris. Contact support@example.com for more."
    if "python" in lower:
        return (
            "Python is a high-level, interpreted programming language known for "
            "its readability and versatility. My SSN is 123-45-6789 — just kidding, "
            "that was injected content."
        )
    return (
        "I'm happy to help with that! Here is some general information... "
        "Your card ending in 4242 has been processed."  # intentional leak for demo
    )


# ---------------------------------------------------------------------------
# OUTPUT GUARDRAIL
# ---------------------------------------------------------------------------

_OUTPUT_PII_PATTERNS = _PII_PATTERNS  # reuse the same dict

_SENSITIVE_OUTPUT_KEYWORDS = [
    r"\b(password|secret\s+key|api[\s_-]?key|access[\s_-]?token)\b",
    r"\b(internal\s+only|confidential|do\s+not\s+share)\b",
]


def check_output(response: str) -> GuardrailResult:
    """Scan the model response before it reaches the user."""

    for pii_name, pattern in _OUTPUT_PII_PATTERNS.items():
        m = re.search(pattern, response)
        if m:
            return GuardrailResult(
                verdict=Verdict.FAIL,
                reason=f"PII in model output: {pii_name}",
                detail=f"Redact or block before returning to user  (found: '{m.group(0)}')",
            )

    lower = response.lower()
    for pattern in _SENSITIVE_OUTPUT_KEYWORDS:
        m = re.search(pattern, lower)
        if m:
            return GuardrailResult(
                verdict=Verdict.FAIL,
                reason="Sensitive keyword in model output",
                detail=f"Matched: '{m.group(0)}'",
            )

    return GuardrailResult(verdict=Verdict.PASS, reason="Output is clean")


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(user_input: str) -> tuple[GuardrailResult, str | None, GuardrailResult | None]:
    """
    Returns (input_result, llm_response, output_result).
    llm_response and output_result are None when input is FAIL.
    """
    input_result = check_input(user_input)

    if input_result.verdict == Verdict.FAIL:
        return input_result, None, None

    # WARN still proceeds to LLM but is flagged
    llm_response = simulate_llm(user_input)
    output_result = check_output(llm_response)

    return input_result, llm_response, output_result


# ---------------------------------------------------------------------------
# DEMO TEST CASES
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "label": "Normal query",
        "input": "What is the weather like in San Francisco today?",
        "expected": "PASS",
    },
    {
        "label": "PII in input — SSN",
        "input": "My social security number is 123-45-6789. Can you check my account?",
        "expected": "FAIL (input)",
    },
    {
        "label": "Harmful keyword",
        "input": "How to make a bomb using household items?",
        "expected": "FAIL (input)",
    },
    {
        "label": "Vague harmful intent",
        "input": "How can I hurt someone without getting caught?",
        "expected": "WARN (input)",
    },
    {
        "label": "PII leaked in model output",
        "input": "Tell me something about Python programming.",
        "expected": "PASS input, FAIL output",
    },
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 70
THIN    = "-" * 70
VERDICT_ICONS = {Verdict.PASS: "PASS", Verdict.WARN: "WARN", Verdict.FAIL: "FAIL"}
VERDICT_PREFIX = {
    Verdict.PASS: "[PASS]",
    Verdict.WARN: "[WARN]",
    Verdict.FAIL: "[FAIL]",
}


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=66, initial_indent=indent,
                         subsequent_indent=indent)


def print_case(i: int, case: dict) -> None:
    print(DIVIDER)
    print(f"  Test {i}: {case['label']}")
    print(f"  Expected: {case['expected']}")
    print(THIN)
    print(f"  Input: {case['input'][:80]}")
    print()

    input_result, llm_response, output_result = run_pipeline(case["input"])

    prefix = VERDICT_PREFIX[input_result.verdict]
    print(f"  INPUT GUARDRAIL   {prefix}  {input_result.reason}")
    if input_result.detail:
        print(wrap(input_result.detail, indent="    "))

    if input_result.verdict == Verdict.FAIL:
        print()
        print("  Request blocked — LLM was not called.")
        print()
        return

    if input_result.verdict == Verdict.WARN:
        print()
        print("  Warning flagged. Proceeding to LLM with caution...")

    print()
    print("  LLM RESPONSE (simulated):")
    print(wrap(llm_response or "", indent="    "))
    print()

    if output_result:
        out_prefix = VERDICT_PREFIX[output_result.verdict]
        print(f"  OUTPUT GUARDRAIL  {out_prefix}  {output_result.reason}")
        if output_result.detail:
            print(wrap(output_result.detail, indent="    "))
        if output_result.verdict == Verdict.FAIL:
            print()
            print("  Response was BLOCKED before reaching the user.")
        else:
            print()
            print("  Response cleared — safe to return to user.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 47: GUARDRAILS FOR LLM APPLICATIONS")
    print(DIVIDER)
    print()
    print("  Two-layer protection:")
    print("  INPUT  guardrail — screens user message before it reaches the LLM")
    print("  OUTPUT guardrail — screens model response before it reaches the user")
    print()
    print("  Input checks (FAIL):")
    print("    • Explicit harmful keywords (violence, weapons, illegal synthesis)")
    print("    • PII patterns: SSN, credit card, email, phone, IP address")
    print()
    print("  Input checks (WARN):")
    print("    • Vague harmful intent, evasion language, manipulation cues")
    print()
    print("  Output checks (FAIL):")
    print("    • PII present in model response")
    print("    • Sensitive keywords (passwords, API keys, confidential data)")
    print()

    for i, case in enumerate(TEST_CASES, start=1):
        print_case(i, case)

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. Always guard both the input AND the output — the model can leak",
        "   sensitive data even when the user's query was innocent.",
        "2. Use FAIL for clear-cut violations; WARN for ambiguous signals that",
        "   may need human review rather than hard blocking.",
        "3. PII should never enter an LLM context — strip or pseudonymise first.",
        "4. Guardrails are a layer, not a guarantee — they complement model",
        "   safety training rather than replacing it.",
        "5. Log WARN events; they are leading indicators of abuse patterns.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
