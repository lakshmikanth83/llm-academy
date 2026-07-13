"""
Topic 02: What is Machine Learning?
====================================
Demo of linear regression trained from scratch via gradient descent.
No external dependencies — pure Python stdlib.
"""

import math


def print_header(title, width=64):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

# (hours_studied, exam_score)
TRAINING_DATA = [
    (1.0, 45),
    (2.0, 50),
    (3.0, 55),
    (4.0, 58),
    (5.0, 65),
    (6.0, 70),
    (7.0, 75),
    (8.0, 80),
    (9.0, 83),
    (10.0, 90),
]


# ---------------------------------------------------------------------------
# LINEAR REGRESSION FROM SCRATCH
# ---------------------------------------------------------------------------

def mean(values):
    return sum(values) / len(values)


def predict(x, weight, bias):
    """y_hat = weight * x + bias"""
    return weight * x + bias


def mean_squared_error(data, weight, bias):
    """MSE = (1/n) * sum((y_hat - y)^2)"""
    n = len(data)
    total = sum((predict(x, weight, bias) - y) ** 2 for x, y in data)
    return total / n


def compute_gradients(data, weight, bias):
    """
    Partial derivatives of MSE with respect to weight and bias.
      dL/dw = (2/n) * sum((y_hat - y) * x)
      dL/db = (2/n) * sum((y_hat - y))
    """
    n = len(data)
    grad_w = (2 / n) * sum((predict(x, weight, bias) - y) * x for x, y in data)
    grad_b = (2 / n) * sum((predict(x, weight, bias) - y)     for x, y in data)
    return grad_w, grad_b


def gradient_descent(data, learning_rate=0.01, iterations=5, w_init=0.0, b_init=0.0):
    """
    Iteratively nudge weight and bias in the direction that reduces error.
    Returns list of (iteration, weight, bias, mse) snapshots.
    """
    w = w_init
    b = b_init
    history = []

    for i in range(iterations):
        mse = mean_squared_error(data, w, b)
        history.append((i + 1, w, b, mse))
        grad_w, grad_b = compute_gradients(data, w, b)
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

    # Record final state
    mse = mean_squared_error(data, w, b)
    history.append((iterations + 1, w, b, mse))
    return history, w, b


# ---------------------------------------------------------------------------
# ANALYTICAL BASELINE (ordinary least squares)
# ---------------------------------------------------------------------------

def ordinary_least_squares(data):
    """
    Closed-form solution for comparison:
      w = cov(x, y) / var(x)
      b = mean(y) - w * mean(x)
    """
    xs = [x for x, _ in data]
    ys = [y for _, y in data]
    x_bar = mean(xs)
    y_bar = mean(ys)
    cov_xy = mean([(x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)])
    var_x  = mean([(x - x_bar) ** 2          for x    in xs])
    w = cov_xy / var_x
    b = y_bar - w * x_bar
    return w, b


# ---------------------------------------------------------------------------
# VISUALISE THE LINE (ASCII)
# ---------------------------------------------------------------------------

def ascii_scatter(data, weight, bias, width=50, height=12):
    xs = [x for x, _ in data]
    ys = [y for _, y in data]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys) - 5, max(ys) + 5

    grid = [[" "] * width for _ in range(height)]

    def col(x):
        return int((x - x_min) / (x_max - x_min) * (width - 1))

    def row(y):
        return height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))

    # Draw the regression line
    for c in range(width):
        x_val = x_min + (x_max - x_min) * c / (width - 1)
        y_val = predict(x_val, weight, bias)
        if y_min <= y_val <= y_max:
            r = row(y_val)
            if 0 <= r < height:
                if grid[r][c] == " ":
                    grid[r][c] = "-"

    # Draw data points (overwrite line)
    for x, y in data:
        r, c = row(y), col(x)
        if 0 <= r < height and 0 <= c < width:
            grid[r][c] = "o"

    print(f"\n  Score ^")
    for r, line in enumerate(grid):
        y_label = y_max - (y_max - y_min) * r / (height - 1)
        print(f"  {y_label:4.0f} | {''.join(line)}")
    print(f"       +{'─' * width}>")
    print(f"       0{' ' * (width // 2 - 2)}Hours studied{' ' * (width // 2 - 6)}10")
    print()
    print("         o = training data point    - = learned regression line")


# ---------------------------------------------------------------------------
# MAIN DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_header("TOPIC 02 — What is Machine Learning?", width=64)
    print(
        "\n  Machine Learning is a subset of AI where a system learns a\n"
        "  mapping from inputs to outputs by adjusting its own parameters\n"
        "  based on data, rather than following hand-written rules.\n"
        "\n  This demo trains a linear regression model to predict exam\n"
        "  scores from hours studied — using only arithmetic."
    )

    # ---- Show the data ----
    print_section("Training Data (10 examples)")
    print(f"  {'Hours Studied':<16} {'Exam Score'}")
    print(f"  {'-'*14:<16} {'-'*10}")
    for x, y in TRAINING_DATA:
        bar = "#" * int(y / 5)
        print(f"  {x:<16.1f} {y:>4}  {bar}")

    # ---- Gradient descent ----
    print_section("Model Parameters Before Training")
    print(
        "  The model starts with initial guesses:\n"
        "    weight (w) = 1.0   (slope: score per extra hour)\n"
        "    bias   (b) = 30.0  (predicted score for 0 hours)\n"
        "    formula: score = w * hours + b"
    )

    LEARNING_RATE = 0.005
    ITERATIONS    = 5

    print_section(f"Gradient Descent — {ITERATIONS} Iterations (learning rate = {LEARNING_RATE})")
    print(
        "  Each iteration:\n"
        "    1. Predict scores for all examples\n"
        "    2. Measure how wrong the predictions are (MSE)\n"
        "    3. Compute which direction reduces the error (gradient)\n"
        "    4. Nudge w and b a small step in that direction\n"
    )

    history, final_w, final_b = gradient_descent(
        TRAINING_DATA,
        learning_rate=LEARNING_RATE,
        iterations=ITERATIONS,
        w_init=1.0,
        b_init=30.0,
    )

    print(f"  {'Iter':<6} {'Weight (w)':<14} {'Bias (b)':<14} {'MSE (error)':<14} Direction")
    print(f"  {'-'*4:<6} {'-'*10:<14} {'-'*8:<14} {'-'*11:<14} {'-'*20}")
    prev_mse = None
    for iter_num, w, b, mse in history:
        if prev_mse is None:
            direction = "  (start)"
        elif mse < prev_mse:
            improvement = prev_mse - mse
            direction = f"  error down {improvement:,.1f}"
        else:
            direction = "  (converged)"
        print(f"  {iter_num:<6} {w:<14.4f} {b:<14.4f} {mse:<14.2f}{direction}")
        prev_mse = mse

    # ---- OLS comparison ----
    ols_w, ols_b = ordinary_least_squares(TRAINING_DATA)
    ols_mse = mean_squared_error(TRAINING_DATA, ols_w, ols_b)

    print_section("Comparison: Gradient Descent vs Exact (Least Squares) Solution")
    print(f"  {'Method':<22} {'weight (w)':<14} {'bias (b)':<14} {'MSE'}")
    print(f"  {'-'*20:<22} {'-'*10:<14} {'-'*8:<14} {'-'*10}")
    print(f"  {'Gradient Descent':<22} {final_w:<14.4f} {final_b:<14.4f} {mean_squared_error(TRAINING_DATA, final_w, final_b):.2f}")
    print(f"  {'Least Squares':<22} {ols_w:<14.4f} {ols_b:<14.4f} {ols_mse:.2f}")
    print(
        "\n  Gradient descent is iterative and approximate.\n"
        "  For linear regression an exact solution exists, but gradient\n"
        "  descent generalises to complex models (neural nets) where no\n"
        "  closed-form solution exists."
    )

    # ---- ASCII chart ----
    print_section("Visualisation (ASCII scatter + regression line)")
    ascii_scatter(TRAINING_DATA, ols_w, ols_b)

    # ---- Predictions ----
    print_section("Predictions on New Inputs")
    print("  Using the learned formula:  score = {:.2f} * hours + {:.2f}\n".format(ols_w, ols_b))
    new_inputs = [3.5, 6.0, 11.0, 0.5]
    print(f"  {'Hours Studied':<16} {'Predicted Score':<18} Notes")
    print(f"  {'-'*14:<16} {'-'*15:<18} {'-'*30}")
    for h in new_inputs:
        pred = predict(h, ols_w, ols_b)
        note = ""
        if h > 10:
            note = "(extrapolation — beyond training range)"
        elif h < 1:
            note = "(extrapolation — below training range)"
        print(f"  {h:<16.1f} {pred:<18.1f} {note}")

    # ---- Key insight ----
    print_section("Key Insight")
    print(
        "  Machine Learning in one sentence:\n"
        "    Start with a model that is completely wrong.\n"
        "    Show it examples. Measure how wrong it is.\n"
        "    Nudge its parameters toward less wrong.\n"
        "    Repeat until it is useful.\n"
        "\n  Everything in ML — from linear regression to GPT-4 — follows\n"
        "  this same loop. What changes is the model architecture,\n"
        "  the error measure, and the scale."
    )

    print("\n" + "=" * 64)
    print("  End of Topic 02 demo.")
    print("=" * 64 + "\n")
