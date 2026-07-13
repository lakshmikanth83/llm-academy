"""
Topic 20: Structured Outputs
=============================
Simulates an LLM extraction pipeline that turns a free-text product review
into a typed JSON object (name, price, rating, pros, cons, sentiment).

A hand-rolled schema validator checks every extraction and reports PASS or
a list of concrete errors — the same feedback loop a real "JSON mode" or
function-calling API gives you. No network calls, no external packages.
"""

from __future__ import annotations

import json
import re
import textwrap

DIVIDER = "=" * 70
THIN = "-" * 70
WRAP_W = 66


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=WRAP_W, initial_indent=indent,
                         subsequent_indent=indent)


def print_json(obj: dict) -> None:
    for line in json.dumps(obj, indent=2).splitlines():
        print(f"    {line}")


def print_validation(errors: list[str]) -> None:
    print("  SCHEMA VALIDATION: PASS" if not errors else "  SCHEMA VALIDATION: FAIL")
    for err in errors:
        print(f"    - {err}")


# ---------------------------------------------------------------------------
# 1. TARGET SCHEMA + LIGHTWEIGHT VALIDATOR
# ---------------------------------------------------------------------------
# In a real system this would be a Pydantic model or a JSON Schema passed to
# the API's "response_format" / tool "input_schema" parameter. Here it is a
# plain dict so the whole demo stays stdlib-only.

PRODUCT_SCHEMA = {
    "required": ["name", "price", "rating", "pros", "cons", "sentiment"],
    "properties": {
        "name":      {"type": "string"},
        "price":     {"type": "number"},
        "rating":    {"type": "number", "minimum": 1, "maximum": 5},
        "pros":      {"type": "array_of_str"},
        "cons":      {"type": "array_of_str"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed"]},
    },
}


def validate_against_schema(data: dict, schema: dict = PRODUCT_SCHEMA) -> list[str]:
    """
    Hand-rolled validator (no jsonschema / pydantic).  Returns a list of
    human-readable error strings; an empty list means the object is valid.
    """
    errors: list[str] = []

    # 1. Required keys must be present.
    for field in schema["required"]:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    # 2. Type / constraint checks for whatever keys ARE present.
    for field, spec in schema["properties"].items():
        if field not in data:
            continue  # already reported as missing above
        value = data[field]
        expected = spec["type"]

        if expected == "string" and not isinstance(value, str):
            errors.append(f"Field '{field}' should be a string, got {type(value).__name__}")
        elif expected == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            errors.append(f"Field '{field}' should be a number, got {type(value).__name__}")
        elif expected == "array_of_str":
            if not isinstance(value, list):
                errors.append(f"Field '{field}' should be a list, got {type(value).__name__}")
            elif not all(isinstance(item, str) for item in value):
                errors.append(f"Field '{field}' should contain only strings")

        # Range check for rating.
        if field == "rating" and isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if lo is not None and hi is not None and not (lo <= value <= hi):
                errors.append(f"Field 'rating' = {value} is out of range [{lo}, {hi}]")

        # Enum check for sentiment.
        if field == "sentiment" and isinstance(value, str) and "enum" in spec:
            if value not in spec["enum"]:
                errors.append(
                    f"Field 'sentiment' = '{value}' is not one of {spec['enum']}"
                )

    return errors


# ---------------------------------------------------------------------------
# 2. SAMPLE FREE-TEXT REVIEWS
# ---------------------------------------------------------------------------

REVIEWS = [
    "The SuperBlender 3000 is fantastic! I bought it for $49.99 and it has "
    "exceeded my expectations. Blending smoothies is fast and the motor is "
    "powerful. I'd give it 5 stars. The only downside is that it's a bit "
    "noisy and the pitcher is hard to clean.",

    "I picked up the AeroFit Running Shoes for $89.00 last week. Comfortable, "
    "lightweight, and great arch support — my feet feel amazing after long "
    "runs. Rating: 4/5. Cons: sizing runs small and the laces come undone "
    "easily.",

    "Bought the NoiseCancel Pro Headphones at $129.99. Sound quality is "
    "excellent and the noise cancellation is superb for flights. However, "
    "battery life is disappointing and they feel cheap. Overall 3 out of 5 "
    "stars — mixed feelings about this purchase.",

    "Terrible experience with the QuickCharge Power Bank, priced at $24.50. "
    "It stopped charging after two days and the build quality feels flimsy. "
    "Customer support was unhelpful. Would not recommend. 1 star.",
]

# A review that an LLM extraction step will fail to fully populate — used to
# demonstrate the validation-and-repair feedback loop.
MALFORMED_REVIEW_TEXT = (
    "I bought the ZenBrew Coffee Maker for $39.99. It leaks water from the "
    "base and takes forever to brew a single cup. Very disappointed with "
    "this purchase."
)


# ---------------------------------------------------------------------------
# 3. "EXTRACTOR" — regex / keyword heuristics standing in for an LLM
# ---------------------------------------------------------------------------
# A real pipeline sends the review text to an LLM with a JSON schema (or a
# Pydantic model) and gets the object back directly. Here plain regex and
# keyword matching stand in for that call so the demo runs offline. These
# heuristics are deliberately imperfect — that is exactly why validation
# matters, both for a real model and for this simulation.

_NAME_CANDIDATE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z]*)(?:\s+(?:[A-Z][A-Za-z]*|\d+))+\b"
)
_LEADING_STOPWORDS = {"the", "i", "overall", "terrible", "bought", "rating", "cons"}

_PRICE_RE = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")

_RATING_PATTERNS = [
    re.compile(r"(\d(?:\.\d)?)\s*/\s*5"),
    re.compile(r"(\d(?:\.\d)?)\s+out\s+of\s+5", re.IGNORECASE),
    re.compile(r"(\d(?:\.\d)?)\s*stars?", re.IGNORECASE),
    re.compile(r"(\d(?:\.\d)?)\s*star\b", re.IGNORECASE),
]

POSITIVE_WORDS = [
    "fantastic", "exceeded", "powerful", "comfortable", "lightweight",
    "amazing", "excellent", "superb", "great", "fast",
]
NEGATIVE_WORDS = [
    "noisy", "hard to clean", "disappointing", "small", "undone", "cheap",
    "flimsy", "unhelpful", "broke", "stopped", "terrible", "poor", "leak",
    "leaks", "forever", "disappointed", "mixed feelings", "not recommend",
]


def extract_name(text: str) -> str | None:
    """Return the longest capitalized multi-word phrase, minus leading stop words."""
    best = None
    for match in _NAME_CANDIDATE_RE.findall(text):
        words = match.split()
        while words and words[0].lower() in _LEADING_STOPWORDS:
            words = words[1:]
        cleaned = " ".join(words)
        if cleaned and (best is None or len(cleaned) > len(best)):
            best = cleaned
    return best


def extract_price(text: str) -> float | None:
    m = _PRICE_RE.search(text)
    return float(m.group(1)) if m else None


def extract_rating(text: str) -> float | None:
    for pattern in _RATING_PATTERNS:
        m = pattern.search(text)
        if m:
            return float(m.group(1))
    return None


def extract_pros_cons(text: str) -> tuple[list[str], list[str]]:
    """Split into rough clauses and bucket each by positive/negative keyword hits."""
    clauses = re.split(r"(?<=[.!])\s+|\s+but\s+|\s+however,?\s+", text)
    pros, cons = [], []
    for clause in clauses:
        clean = clause.strip().rstrip(".!")
        if not clean:
            continue
        lower = clean.lower()
        has_neg = any(w in lower for w in NEGATIVE_WORDS)
        has_pos = any(w in lower for w in POSITIVE_WORDS)
        if has_neg and not has_pos:
            cons.append(clean)
        elif has_pos and not has_neg:
            pros.append(clean)
    return pros, cons


def extract_sentiment(pros: list[str], cons: list[str]) -> str:
    if pros and cons:
        return "mixed"
    if pros:
        return "positive"
    if cons:
        return "negative"
    return "mixed"  # ambiguous — no strong signal either way


def extract_review(text: str) -> dict:
    """
    The simulated 'structured output' step. A real LLM call would return this
    dict directly (already JSON); here it is assembled from regex heuristics.
    Fields the extractor could not find are OMITTED — exactly what happens
    when an LLM skips a field it wasn't confident about.
    """
    name = extract_name(text)
    price = extract_price(text)
    rating = extract_rating(text)
    pros, cons = extract_pros_cons(text)
    sentiment = extract_sentiment(pros, cons)

    data = {}
    if name is not None:
        data["name"] = name
    if price is not None:
        data["price"] = price
    if rating is not None:
        data["rating"] = rating
    data["pros"] = pros
    data["cons"] = cons
    data["sentiment"] = sentiment
    return data


# ---------------------------------------------------------------------------
# 4. DEMO — extract + validate each review
# ---------------------------------------------------------------------------

def run_extraction_demo(review_text: str, index: int) -> None:
    print(DIVIDER)
    print(f"  REVIEW {index}")
    print(DIVIDER)
    print()
    print("  Raw free text:")
    print(wrap(review_text))
    print()

    extracted = extract_review(review_text)
    print("  Extracted structured output (simulated LLM response):")
    print_json(extracted)
    print()
    print_validation(validate_against_schema(extracted))
    print()


# ---------------------------------------------------------------------------
# 5. DEMO — validation failure + repair loop
# ---------------------------------------------------------------------------

def run_repair_demo() -> None:
    print(DIVIDER)
    print("  VALIDATION -> REPAIR LOOP  (a deliberately incomplete extraction)")
    print(DIVIDER)
    print()
    print("  Raw free text:")
    print(wrap(MALFORMED_REVIEW_TEXT))
    print()

    broken = extract_review(MALFORMED_REVIEW_TEXT)
    print("  Extracted structured output (rating heuristic found no match):")
    print_json(broken)
    print()
    print_validation(validate_against_schema(broken))
    print()

    print(wrap(
        "In a real pipeline, these errors would be fed back to the model in "
        "a follow-up prompt: 'Your previous response was missing the rating "
        "field — please try again including it.' Here we repair it directly "
        "to show the resulting object becomes valid:"
    ))
    print()

    repaired = dict(broken)
    repaired["rating"] = 1.0  # inferred from the clearly negative review text
    print("  Repaired structured output:")
    print_json(repaired)
    print()
    print_validation(validate_against_schema(repaired))
    print()


# ---------------------------------------------------------------------------
# 6. WHY STRUCTURED OUTPUT MATTERS — free-form vs JSON contrast
# ---------------------------------------------------------------------------

def run_contrast_demo() -> None:
    print(DIVIDER)
    print("  WHY STRUCTURED OUTPUT MATTERS")
    print(DIVIDER)
    print()

    review = REVIEWS[0]
    free_form_answer = (
        "The product is called SuperBlender 3000 and it costs around fifty "
        "dollars. Users seem to like it a lot — I'd say roughly five stars. "
        "It's great for smoothies, though it can be a little loud and the "
        "pitcher is annoying to wash."
    )

    print("  Same review, answered two different ways by an LLM:")
    print()
    print("  (A) FREE-FORM PARAGRAPH ANSWER")
    print(THIN)
    print(wrap(free_form_answer))
    print()
    print(wrap(
        "Downstream code must now guess: is the price '$50' or '$49.99'? Is "
        "the rating exactly 5, or 'roughly five'? Which sentence is a pro and "
        "which is a con? Every consumer of this text needs its own fragile "
        "parsing logic — and it will break on the next review."
    ))
    print()

    print("  (B) STRUCTURED JSON ANSWER")
    print(THIN)
    print_json(extract_review(review))
    print()
    print(wrap(
        "Downstream code just does data['price'], data['rating'], "
        "data['sentiment'] — no parsing, no ambiguity, no guessing. The "
        "contract between the LLM and the rest of the application is the "
        "schema, not the prose."
    ))
    print()


# ---------------------------------------------------------------------------
# 7. COMPARISON — approaches to getting structured data from an LLM
# ---------------------------------------------------------------------------

def run_comparison_table() -> None:
    print(DIVIDER)
    print("  COMPARISON: APPROACHES TO STRUCTURED EXTRACTION")
    print(DIVIDER)
    print()

    rows = [
        ("JSON mode",
         "Valid JSON syntax",
         "Low (one API flag)",
         "Fields can still be missing/wrong type"),
        ("Function-calling / tool schema",
         "Matches schema: fields, types, enums",
         "Medium (define schema once)",
         "Model may misread ambiguous input"),
        ("Regex parsing of free text",
         "No guarantee at all",
         "High (brittle per format)",
         "Breaks the moment wording changes"),
    ]

    col1, col2, col3, col4 = 32, 38, 29, 40
    header = (
        f"  {'Approach':<{col1}} {'Guarantee':<{col2}} "
        f"{'Setup effort':<{col3}} {'Main failure mode':<{col4}}"
    )
    print(header)
    print("  " + "-" * (col1 + col2 + col3 + col4))
    for approach, guarantee, effort, failure in rows:
        print(f"  {approach:<{col1}} {guarantee:<{col2}} {effort:<{col3}} {failure:<{col4}}")
    print()
    print(
        "  Note: this demo's 'extractor' IS the regex-parsing row — it stands\n"
        "  in for an LLM so the script can run offline. A real deployment\n"
        "  would replace it with a JSON-mode or function-calling API call,\n"
        "  which is why we still validate the result before trusting it."
    )
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 20: STRUCTURED OUTPUTS — OFFLINE EXTRACTION DEMO")
    print(DIVIDER)
    print()
    print("  Pipeline overview")
    print(THIN)
    stages = [
        ("Schema",     "Plain dict describing required fields, types, enums"),
        ("Extractor",  "Regex/keyword heuristics simulate an LLM's JSON reply"),
        ("Validator",  "Hand-rolled checker — no jsonschema / pydantic needed"),
        ("Repair",     "Feed validation errors back and re-check the fix"),
    ]
    for name, desc in stages:
        print(f"  {name:<12}  {desc}")
    print()
    print(f"  Sample reviews to process: {len(REVIEWS)}")
    print()

    for i, review in enumerate(REVIEWS, start=1):
        run_extraction_demo(review, i)

    run_repair_demo()
    run_contrast_demo()
    run_comparison_table()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. A schema turns an LLM's free-text answer into a typed object that",
        "   downstream code can consume with zero string parsing.",
        "2. JSON mode only guarantees valid syntax — it does NOT guarantee the",
        "   fields, types, or enum values you actually need.",
        "3. Always validate structured output against your schema before",
        "   trusting it; treat every extraction as 'untrusted until checked'.",
        "4. When validation fails, feed the specific errors back to the model",
        "   (or, as shown here, repair the object) and re-validate.",
        "5. Function-calling / tool schemas give stronger guarantees than",
        "   bare JSON mode; hand-written regex parsing gives the weakest.",
        "6. The schema is the real contract between the LLM and your app —",
        "   not the wording of the prompt.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
