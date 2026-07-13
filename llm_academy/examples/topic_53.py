"""
Topic 53: Fine-tuning a Small Language Model
==============================================
Demo simulating LoRA fine-tuning of a small language model.
This does NOT train a real model — it numerically simulates training and
validation loss curves as functions of LoRA rank, learning rate, and epoch
count, since real fine-tuning needs GPU/ML libraries this app doesn't use.
"""

import json
import math
import random
import textwrap

DIVIDER = "=" * 70
THIN = "-" * 70


def print_header(title, width=70):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title):
    print(f"\n--- {title} ---")


def wrap(text, indent="  "):
    return textwrap.fill(text, width=66, initial_indent=indent,
                          subsequent_indent=indent)


# ---------------------------------------------------------------------------
# SECTION 1: THE TRAINING DATASET (JSONL)
# ---------------------------------------------------------------------------
# Task: teach a small model to turn a free-form customer message into a
# terse, strictly-formatted JSON triage record. This is a classic
# instruction fine-tuning use case — the base model already "knows" English,
# but we want it to reliably emit one exact output shape every time.

TRAINING_DATA = [
    {
        "prompt": "My order #4471 never arrived and it's been 2 weeks.",
        "completion": '{"intent":"shipping_issue","priority":"high","response":"Sorry about the delay on order #4471 — escalating to logistics now."}',
    },
    {
        "prompt": "How do I reset my account password?",
        "completion": '{"intent":"account_help","priority":"low","response":"Go to Settings > Security > Reset Password and follow the emailed link."}',
    },
    {
        "prompt": "I was charged twice for the same subscription this month.",
        "completion": '{"intent":"billing_issue","priority":"high","response":"We see the duplicate charge and are refunding it within 3 business days."}',
    },
    {
        "prompt": "Can you tell me if you ship to New Zealand?",
        "completion": '{"intent":"shipping_question","priority":"low","response":"Yes, we ship to New Zealand with delivery in 7-10 business days."}',
    },
    {
        "prompt": "The app crashes every time I try to upload a photo.",
        "completion": '{"intent":"bug_report","priority":"medium","response":"Thanks for the report — forwarding to engineering with your device info."}',
    },
    {
        "prompt": "I want to cancel my plan immediately, this is unacceptable.",
        "completion": '{"intent":"cancellation","priority":"high","response":"Your plan has been cancelled effective today, no further charges will apply."}',
    },
    {
        "prompt": "Do you offer a student discount?",
        "completion": '{"intent":"pricing_question","priority":"low","response":"Yes, students get 20% off with a valid .edu email at checkout."}',
    },
    {
        "prompt": "My package arrived damaged, the box was crushed.",
        "completion": '{"intent":"shipping_issue","priority":"medium","response":"Sorry to hear that — a replacement is being shipped at no extra cost."}',
    },
    {
        "prompt": "Is there an API I can use to integrate with your service?",
        "completion": '{"intent":"technical_question","priority":"low","response":"Yes, our REST API docs are at developer.example.com/docs."}',
    },
    {
        "prompt": "I think my account was hacked, I see logins from another country.",
        "completion": '{"intent":"security_issue","priority":"high","response":"Locking your account now and sending a password-reset link immediately."}',
    },
]


def print_dataset_section():
    print_section("SECTION 1: The Fine-tuning Dataset (JSONL)")
    print(wrap(
        "Fine-tuning teaches an already-capable base model a narrow, "
        "consistent behaviour: always turn a customer message into a "
        "strict JSON triage record with 'intent', 'priority', and "
        "'response' fields. The data below is what would actually be "
        "saved to train.jsonl — one self-contained JSON object per line."
    ))
    print()
    print(f"  {len(TRAINING_DATA)} training examples "
          f"({int(len(TRAINING_DATA) * 0.8)} train / "
          f"{len(TRAINING_DATA) - int(len(TRAINING_DATA) * 0.8)} held out for validation):")
    print(THIN)
    for row in TRAINING_DATA:
        print("  " + json.dumps(row, separators=(",", ":")))
    print(THIN)
    print(wrap(
        "Notice every completion follows the exact same JSON shape. "
        "Consistency in the target format matters more than dataset size — "
        "a model can learn a rigid output pattern from a few hundred clean "
        "examples far more reliably than from thousands of inconsistent ones."
    ))


# ---------------------------------------------------------------------------
# SECTION 2: WHAT LoRA ACTUALLY TRAINS
# ---------------------------------------------------------------------------

def estimate_lora_params(base_params, hidden_dim, num_target_modules, rank):
    """
    Rough estimate of trainable parameters for a LoRA adapter.

    Each target module (e.g. a q_proj or v_proj linear layer of shape
    hidden_dim x hidden_dim) gets two low-rank matrices:
      A: hidden_dim x rank
      B: rank x hidden_dim
    Trainable params per module = 2 * hidden_dim * rank.
    """
    per_module = 2 * hidden_dim * rank
    total_lora_params = per_module * num_target_modules
    pct_of_base = 100.0 * total_lora_params / base_params
    return total_lora_params, pct_of_base


def print_lora_section():
    print_section("SECTION 2: LoRA — Training a Tiny Adapter, Not the Whole Model")
    print(wrap(
        "Full fine-tuning updates every one of a model's weights. LoRA "
        "(Low-Rank Adaptation) freezes the base model entirely and injects "
        "small trainable matrices next to a few target layers (usually the "
        "attention projections: q_proj, k_proj, v_proj, o_proj). Only those "
        "small matrices are trained."
    ))
    print()
    print("  Key hyperparameters:")
    print("    r (rank)        - width of the low-rank matrices; higher r = more capacity")
    print("    alpha           - scaling factor applied to the LoRA update (effective LR knob)")
    print("    target_modules  - which layers get an adapter (e.g. q_proj, v_proj)")
    print()

    base_params = 1_200_000_000  # a 1.2B-parameter small model
    hidden_dim = 2048
    num_target_modules = 4  # q_proj, k_proj, v_proj, o_proj

    print(f"  Illustrative example — base model: {base_params:,} params, "
          f"hidden_dim={hidden_dim}, {num_target_modules} target modules")
    print(f"  {'rank (r)':<10} {'trainable params':<20} {'% of base model':<18}")
    print(f"  {'-'*8:<10} {'-'*18:<20} {'-'*16:<18}")
    for rank in (4, 8, 16, 32, 64):
        n_params, pct = estimate_lora_params(base_params, hidden_dim, num_target_modules, rank)
        print(f"  {rank:<10} {n_params:<20,} {pct:<18.3f}")

    print()
    print(wrap(
        "Even at r=64, the adapter trains well under 1% of the base model's "
        "parameters. This is why LoRA fine-tuning fits on consumer GPUs and "
        "why swapping adapters is cheap — you ship a few megabytes instead "
        "of re-shipping the whole model."
    ))


# ---------------------------------------------------------------------------
# SECTION 3: THE TRAINING-LOOP SIMULATOR
# ---------------------------------------------------------------------------
# This is a *numeric model* of training dynamics, not a real model. Loss
# starts high and decays exponentially toward a floor. The floor depends on
# LoRA rank (more capacity -> can fit the data more tightly -> lower floor).
# Validation loss tracks training loss early on, but an "overfitting
# pressure" term grows with rank, epoch, and an overly aggressive learning
# rate, pulling validation loss upward after enough epochs — modelling the
# real-world symptom of memorizing training data instead of generalizing.

def simulate_training(rank, learning_rate, epochs, seed):
    """
    Returns a list of (epoch, train_loss, val_loss) tuples.
    Deterministic given `seed`.
    """
    random.seed(seed)

    start_loss = 4.0
    # More capacity (higher rank) lets the model fit training data more
    # tightly, so the achievable training-loss floor is lower.
    train_floor = max(0.05, 0.55 - 0.03 * rank)
    # How fast loss decays depends on the learning rate.
    decay_rate = learning_rate * 40.0

    history = []
    for epoch in range(1, epochs + 1):
        train_noise = random.gauss(0, 0.02)
        train_loss = train_floor + (start_loss - train_floor) * math.exp(-decay_rate * epoch)
        train_loss = max(0.02, train_loss + train_noise)

        # Overfitting pressure: grows with rank (capacity to memorize),
        # with how far into training we are, and is amplified by an
        # aggressive learning rate that overshoots the generalizable minimum.
        progress = epoch / epochs
        lr_overshoot = max(0.0, learning_rate - 0.0015) * 300.0
        overfit_pressure = (rank / 64.0) * (progress ** 2) * (0.4 + lr_overshoot)

        val_floor = train_floor + 0.08
        val_noise = random.gauss(0, 0.03)
        val_loss = (val_floor + (start_loss - val_floor) * math.exp(-decay_rate * 0.8 * epoch)
                    + overfit_pressure + val_noise)
        val_loss = max(0.02, val_loss)

        history.append((epoch, train_loss, val_loss))
    return history


def sparkline(values, width_chars="▁▂▃▄▅▆▇█"):
    """Render a list of numbers as a compact block-character sparkline."""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return width_chars[-1] * len(values)
    chars = []
    for v in values:
        norm = (v - lo) / span
        idx = min(len(width_chars) - 1, int(norm * (len(width_chars) - 1)))
        chars.append(width_chars[idx])
    return "".join(chars)


# ---------------------------------------------------------------------------
# SECTION 4: RUNNING THE SIMULATOR ACROSS HYPERPARAMETER CONFIGS
# ---------------------------------------------------------------------------

CONFIGS = [
    {"name": "Config A: r=4,  lr=0.0005 (conservative)", "rank": 4, "lr": 0.0005, "seed": 1},
    {"name": "Config B: r=16, lr=0.0015 (balanced)", "rank": 16, "lr": 0.0015, "seed": 2},
    {"name": "Config C: r=64, lr=0.0060 (aggressive)", "rank": 64, "lr": 0.0060, "seed": 3},
]
EPOCHS = 10


def print_training_table(name, history):
    print(f"  {name}")
    print(f"  {'epoch':<7} {'train_loss':<12} {'val_loss':<12} {'gap':<8}")
    print(f"  {'-'*5:<7} {'-'*10:<12} {'-'*8:<12} {'-'*6:<8}")
    for epoch, train_loss, val_loss in history:
        gap = val_loss - train_loss
        print(f"  {epoch:<7} {train_loss:<12.4f} {val_loss:<12.4f} {gap:<8.4f}")

    train_vals = [t for _, t, _ in history]
    val_vals = [v for _, _, v in history]
    print(f"  train sparkline: {sparkline(train_vals)}   (low = short bar)")
    print(f"  val   sparkline: {sparkline(val_vals)}   (low = short bar)")
    print()


def run_configs():
    results = []
    for cfg in CONFIGS:
        history = simulate_training(cfg["rank"], cfg["lr"], EPOCHS, cfg["seed"])
        results.append((cfg, history))
        print_training_table(cfg["name"], history)
    return results


def print_summary(results):
    print_section("SECTION 5: Comparing the Runs")
    print(f"  {'config':<42} {'final train':<13} {'final val':<12} {'gap':<8}")
    print(f"  {'-'*38:<42} {'-'*11:<13} {'-'*10:<12} {'-'*6:<8}")
    for cfg, history in results:
        _, final_train, final_val = history[-1]
        gap = final_val - final_train
        print(f"  {cfg['name']:<42} {final_train:<13.4f} {final_val:<12.4f} {gap:<8.4f}")

    print()
    print(wrap(
        "Config A (small rank, small learning rate) converges slowly but "
        "train and validation loss stay close together — little capacity "
        "to overfit. Config B finds a good middle ground: low final loss "
        "with a small generalization gap. Config C reaches the lowest "
        "training loss of all three, but its validation loss visibly rises "
        "in later epochs — high rank gave it enough capacity to memorize "
        "training examples, and the aggressive learning rate accelerated "
        "that memorization instead of general learning. More capacity and "
        "a faster learning rate are not automatically better."
    ))


# ---------------------------------------------------------------------------
# SECTION 6: FULL FINE-TUNING vs LoRA vs QLoRA
# ---------------------------------------------------------------------------

def print_method_comparison():
    print_section("SECTION 6: Full Fine-tuning vs LoRA vs QLoRA")
    rows = [
        ("Trainable params", "100% of model", "~0.1-1% of model", "~0.1-1% of model"),
        ("Base model precision", "fp16/bf16", "fp16/bf16", "4-bit quantized"),
        ("GPU memory (7B model)", "~60-80GB", "~16-20GB", "~6-8GB"),
        ("Output quality", "Highest ceiling", "Very close to full FT", "Slightly below LoRA"),
        ("Typical use case", "Large budget, max quality", "Consumer/prosumer GPU tuning", "Single consumer GPU, laptops"),
    ]
    col0, col1, col2, col3 = 22, 22, 22, 22
    print(f"  {'Aspect':<{col0}} {'Full Fine-tuning':<{col1}} {'LoRA':<{col2}} {'QLoRA':<{col3}}")
    print("  " + "-" * (col0 + col1 + col2 + col3))
    for aspect, full, lora, qlora in rows:
        print(f"  {aspect:<{col0}} {full:<{col1}} {lora:<{col2}} {qlora:<{col3}}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print_header("TOPIC 53 — Fine-tuning a Small Language Model", width=70)
    print(wrap(
        "This demo simulates the full workflow of LoRA fine-tuning a small "
        "language model: preparing a JSONL dataset, understanding what LoRA "
        "actually trains, and running a numeric simulation of training "
        "dynamics across different hyperparameter choices. No GPU or ML "
        "library is used — training and validation loss are modelled with "
        "an explicit decay-plus-overfitting formula."
    ))

    print_dataset_section()
    print_lora_section()

    print_section("SECTION 3/4: Simulated Training Runs (10 epochs each)")
    print(wrap(
        "Three hyperparameter configurations are run below. Loss is computed "
        "each epoch from a formula: an exponential decay toward a rank-"
        "dependent floor, plus Gaussian noise, plus an overfitting-pressure "
        "term for validation loss that grows with rank, training progress, "
        "and an overly high learning rate."
    ))
    print()
    results = run_configs()

    print_summary(results)
    print_method_comparison()

    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. LoRA freezes the base model and trains small low-rank adapter",
        "   matrices — often under 1% of total parameters — making fine-tuning",
        "   feasible on a single consumer GPU.",
        "2. A consistent, clean JSONL dataset matters more than dataset size;",
        "   the model learns the output *shape* from repeated exact patterns.",
        "3. Higher LoRA rank gives more capacity to fit training data (lower",
        "   training loss) but also more capacity to overfit it.",
        "4. Learning rate that is too aggressive accelerates memorization,",
        "   not generalization — watch the train/val gap, not just train loss.",
        "5. Rising validation loss alongside falling training loss is the",
        "   textbook overfitting signature; fix it with fewer epochs, less",
        "   rank, more data, or a lower learning rate — not more training.",
        "6. QLoRA trades a small quality dip for running fine-tuning on GPUs",
        "   with as little as 6-8GB of memory by quantizing the base model.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
