"""
Topic 01: What is AI?
=====================
Demo comparing rule-based AI vs learning-based AI.
No external dependencies — pure Python stdlib.
"""

import math
from collections import defaultdict


def print_header(title, width=60):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# DEMO 1: Rule-Based Classifier
# ---------------------------------------------------------------------------

def rule_based_weather_recommendation(temperature_c, is_raining, wind_kmh):
    """
    Explicitly coded rules — every decision is hand-written by a programmer.
    """
    if is_raining:
        if temperature_c < 5:
            return "Wear a heavy coat and carry an umbrella."
        elif temperature_c < 15:
            return "Wear a jacket and carry an umbrella."
        else:
            return "Carry an umbrella; light clothing is fine."
    else:
        if temperature_c < 0:
            return "Wear a heavy winter coat, gloves, and a scarf."
        elif temperature_c < 10:
            return "Wear a warm jacket."
        elif temperature_c < 20:
            if wind_kmh > 30:
                return "Wear a light jacket — it's breezy."
            return "A light jacket or hoodie should work."
        elif temperature_c < 30:
            return "T-shirt weather. Enjoy!"
        else:
            return "It's hot — wear light clothing and stay hydrated."


def demo_rule_based():
    print_section("DEMO 1: Rule-Based AI (Hand-Coded Rules)")
    print(
        "  A programmer writes every rule explicitly.\n"
        "  The system cannot learn — it only knows what the rules say.\n"
    )

    test_cases = [
        (28, False, 10),
        (-3, False, 5),
        (12, True,  20),
        (4,  True,  40),
        (22, False, 45),
    ]

    print(f"  {'Temp(°C)':<10} {'Rain':<6} {'Wind(km/h)':<12} Recommendation")
    print(f"  {'-'*8:<10} {'-'*4:<6} {'-'*10:<12} {'-'*30}")
    for temp, rain, wind in test_cases:
        rec = rule_based_weather_recommendation(temp, rain, wind)
        rain_str = "Yes" if rain else "No"
        print(f"  {temp:<10} {rain_str:<6} {wind:<12} {rec}")

    print(
        "\n  How it works:\n"
        "    if is_raining and temperature < 5:  -> 'Heavy coat + umbrella'\n"
        "    if temperature > 28:                -> 'T-shirt weather'\n"
        "    ... (every rule manually written)\n"
        "\n  Limitation: Adding a new condition (e.g. humidity) requires\n"
        "  editing code. Cannot improve from experience."
    )


# ---------------------------------------------------------------------------
# DEMO 2: Learning-Based Classifier (Frequency Counting)
# ---------------------------------------------------------------------------

class LearnedWeatherClassifier:
    """
    Learns associations from labeled examples using frequency counting.
    No rules are written by hand — patterns emerge from data.
    """

    def __init__(self):
        # label -> list of feature tuples seen during training
        self._examples = defaultdict(list)
        self._labels = []

    def _bucketize(self, temperature_c, is_raining, wind_kmh):
        """Convert raw features into discrete buckets for counting."""
        if temperature_c < 5:
            temp_bucket = "freezing"
        elif temperature_c < 15:
            temp_bucket = "cold"
        elif temperature_c < 25:
            temp_bucket = "mild"
        else:
            temp_bucket = "hot"

        rain_bucket = "rainy" if is_raining else "dry"

        if wind_kmh < 20:
            wind_bucket = "calm"
        elif wind_kmh < 40:
            wind_bucket = "breezy"
        else:
            wind_bucket = "windy"

        return (temp_bucket, rain_bucket, wind_bucket)

    def train(self, examples):
        """
        examples: list of (temperature_c, is_raining, wind_kmh, label)
        """
        for temp, rain, wind, label in examples:
            features = self._bucketize(temp, rain, wind)
            self._examples[label].append(features)
            if label not in self._labels:
                self._labels.append(label)

    def predict(self, temperature_c, is_raining, wind_kmh):
        """
        Find the label whose training examples best match the new input
        by counting feature overlaps (naive similarity).
        """
        query = self._bucketize(temperature_c, is_raining, wind_kmh)
        best_label = None
        best_score = -1

        for label, feature_list in self._examples.items():
            total_matches = 0
            for stored in feature_list:
                matches = sum(q == s for q, s in zip(query, stored))
                total_matches += matches
            avg_score = total_matches / len(feature_list)
            if avg_score > best_score:
                best_score = avg_score
                best_label = label

        return best_label, query


def demo_learning_based():
    print_section("DEMO 2: Learning-Based AI (Learned from Data)")
    print(
        "  No rules are coded by hand. The system observes labeled examples\n"
        "  and extracts patterns on its own.\n"
    )

    training_data = [
        # (temp_c, raining, wind_kmh, label)
        (30, False, 5,  "T-shirt and sunscreen"),
        (28, False, 8,  "T-shirt and sunscreen"),
        (25, False, 15, "Light clothing"),
        (20, False, 10, "Light jacket"),
        (18, False, 35, "Light jacket — it is breezy"),
        (10, False, 20, "Warm jacket"),
        (5,  False, 10, "Heavy coat"),
        (-2, False, 5,  "Heavy coat, gloves, scarf"),
        (15, True,  10, "Jacket and umbrella"),
        (8,  True,  20, "Jacket and umbrella"),
        (3,  True,  10, "Heavy coat and umbrella"),
        (22, True,  5,  "Umbrella only"),
    ]

    clf = LearnedWeatherClassifier()

    print("  Training examples shown to the model:")
    print(f"  {'Temp(°C)':<10} {'Rain':<6} {'Wind(km/h)':<12} Label")
    print(f"  {'-'*8:<10} {'-'*4:<6} {'-'*10:<12} {'-'*30}")
    for temp, rain, wind, label in training_data:
        clf.train([(temp, rain, wind, label)])
        rain_str = "Yes" if rain else "No"
        print(f"  {temp:<10} {rain_str:<6} {wind:<12} {label}")

    print("\n  Now predicting on new inputs the model has never seen:")
    test_cases = [
        (27, False, 12),
        (-1, False, 8),
        (13, True,  15),
        (21, True,  6),
    ]

    print(f"\n  {'Temp(°C)':<10} {'Rain':<6} {'Wind(km/h)':<12} {'Buckets':<30} Prediction")
    print(f"  {'-'*8:<10} {'-'*4:<6} {'-'*10:<12} {'-'*28:<30} {'-'*30}")
    for temp, rain, wind in test_cases:
        label, buckets = clf.predict(temp, rain, wind)
        rain_str = "Yes" if rain else "No"
        bucket_str = f"({buckets[0]}, {buckets[1]}, {buckets[2]})"
        print(f"  {temp:<10} {rain_str:<6} {wind:<12} {bucket_str:<30} {label}")

    print(
        "\n  How it learned:\n"
        "    The model never received rules like 'if raining: use umbrella'.\n"
        "    It discovered that 'rainy' examples map to umbrella-related\n"
        "    labels purely by counting feature co-occurrences in training data."
    )


# ---------------------------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------------------------

def demo_comparison():
    print_section("COMPARISON: Rule-Based vs Learning-Based AI")

    rows = [
        ("Knowledge source",  "Rules written by programmer",   "Patterns extracted from data"),
        ("Adaptability",      "Requires code change",          "Retrain on new examples"),
        ("Transparency",      "Fully readable if/else logic",  "Opaque — lives in statistics"),
        ("Handles new cases", "Only if a rule covers it",      "Generalizes to unseen inputs"),
        ("Scales with data",  "No benefit from more data",     "More data -> better patterns"),
        ("Expertise needed",  "Domain expert writes rules",    "Labeled examples needed"),
    ]

    col1, col2, col3 = 22, 34, 34
    header = f"  {'Aspect':<{col1}} {'Rule-Based':<{col2}} {'Learning-Based':<{col3}}"
    print(header)
    print("  " + "-" * (col1 + col2 + col3))
    for aspect, rule, learn in rows:
        print(f"  {aspect:<{col1}} {rule:<{col2}} {learn:<{col3}}")

    print(
        "\n  Key insight:\n"
        "  Rule-based systems are predictable and explainable but brittle.\n"
        "  Learning-based systems are flexible and scalable but require data.\n"
        "  Modern AI (like LLMs) is learning-based — trained on billions of\n"
        "  examples, not billions of hand-coded rules."
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_header("TOPIC 01 — What is AI?", width=60)
    print(
        "\n  Artificial Intelligence is the field of building systems that\n"
        "  exhibit intelligent behaviour. There are two broad approaches:\n"
        "\n    1. Rule-Based AI  — humans encode knowledge as explicit rules.\n"
        "    2. Learning-Based  — the system extracts rules from examples.\n"
        "\n  Both are demonstrated below using a weather-recommendation task."
    )

    demo_rule_based()
    demo_learning_based()
    demo_comparison()

    print("\n" + "=" * 60)
    print("  End of Topic 01 demo.")
    print("=" * 60 + "\n")
