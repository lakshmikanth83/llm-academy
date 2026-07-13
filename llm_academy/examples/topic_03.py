"""
Topic 03: Neural Networks
==========================
Builds a tiny multi-layer perceptron (MLP) completely from scratch — no
numpy, no frameworks, just lists of floats and `for` loops — and trains it
with manual backpropagation on a toy 2D XOR-style classification problem.
Three architectures (no hidden layer, one hidden layer, two hidden layers)
are compared to show how depth and non-linear activations reshape the
decision boundary.
"""

import math
import random

DIVIDER = "=" * 70
THIN = "-" * 70


# ---------------------------------------------------------------------------
# 1. ACTIVATION FUNCTIONS
# ---------------------------------------------------------------------------
# Without a non-linear activation, stacking linear layers is pointless —
# W2 * (W1 * x + b1) + b2 is still just a linear function of x. The
# activation is what lets a network bend space and separate non-linear
# patterns like XOR.

def sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def relu(z):
    return z if z > 0.0 else 0.0


def activate(z, kind):
    if kind == "sigmoid":
        return sigmoid(z)
    if kind == "relu":
        return relu(z)
    if kind == "linear":
        return z
    raise ValueError(f"Unknown activation: {kind}")


def activate_deriv(z, a, kind):
    """d(activation)/dz, expressed using whichever of (z, a) is cheaper."""
    if kind == "sigmoid":
        return a * (1.0 - a)
    if kind == "relu":
        return 1.0 if z > 0.0 else 0.0
    if kind == "linear":
        return 1.0
    raise ValueError(f"Unknown activation: {kind}")


# ---------------------------------------------------------------------------
# 2. DENSE LAYER — weights, biases, forward pass, backward pass
# ---------------------------------------------------------------------------

class DenseLayer:
    """A fully-connected layer: a = activation(W . x + b)."""

    def __init__(self, n_inputs, n_outputs, activation, rng):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        # Small random weights, scaled by 1/sqrt(n_inputs) so pre-activation
        # values start near zero and don't saturate the activation early.
        scale = 1.0 / math.sqrt(n_inputs)
        self.weights = [
            [rng.uniform(-1.0, 1.0) * scale for _ in range(n_inputs)]
            for _ in range(n_outputs)
        ]
        self.biases = [0.0] * n_outputs
        # Cached from the most recent forward() call — needed by backward().
        self._last_input = None
        self._z = None
        self._a = None

    def forward(self, x):
        self._last_input = x
        self._z = [
            self.biases[i] + sum(w * xi for w, xi in zip(self.weights[i], x))
            for i in range(self.n_outputs)
        ]
        self._a = [activate(z, self.activation) for z in self._z]
        return self._a

    def backward(self, d_a, learning_rate):
        """
        d_a: dL/d(this layer's output), one value per output neuron.
        Updates weights/biases in place via gradient descent and returns
        dL/d(this layer's input) so the previous layer can keep backprop-ing.
        """
        d_z = [
            d_a[i] * activate_deriv(self._z[i], self._a[i], self.activation)
            for i in range(self.n_outputs)
        ]

        d_input = [
            sum(d_z[i] * self.weights[i][j] for i in range(self.n_outputs))
            for j in range(self.n_inputs)
        ]

        for i in range(self.n_outputs):
            for j in range(self.n_inputs):
                self.weights[i][j] -= learning_rate * d_z[i] * self._last_input[j]
            self.biases[i] -= learning_rate * d_z[i]

        return d_input

    def param_count(self):
        return self.n_inputs * self.n_outputs + self.n_outputs


# ---------------------------------------------------------------------------
# 3. THE NETWORK — stacks DenseLayers, runs forward/backward, predicts
# ---------------------------------------------------------------------------

class MLP:
    """A minimal multi-layer perceptron trained with manual backpropagation."""

    def __init__(self, layer_sizes, activations, seed):
        assert len(activations) == len(layer_sizes) - 1
        rng = random.Random(seed)
        self.layers = [
            DenseLayer(layer_sizes[i], layer_sizes[i + 1], activations[i], rng)
            for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, x):
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def train_step(self, x, y_true, learning_rate):
        """One example: forward pass, binary cross-entropy loss, backprop."""
        y_pred = self.forward(x)[0]
        eps = 1e-9
        p = min(max(y_pred, eps), 1.0 - eps)  # clip to avoid log(0)
        loss = -(y_true * math.log(p) + (1.0 - y_true) * math.log(1.0 - p))
        # dL/da for the single output neuron (BCE derivative w.r.t. its
        # activated output). activate_deriv() then converts this to dL/dz.
        d_a = [(p - y_true) / (p * (1.0 - p))]
        for layer in reversed(self.layers):
            d_a = layer.backward(d_a, learning_rate)
        return loss

    def predict_prob(self, x):
        return self.forward(x)[0]

    def predict(self, x):
        return 1 if self.predict_prob(x) >= 0.5 else 0

    def param_count(self):
        return sum(layer.param_count() for layer in self.layers)


# ---------------------------------------------------------------------------
# 4. TOY DATASET — the XOR pattern (classic non-linearly-separable problem)
# ---------------------------------------------------------------------------

def make_xor_dataset(points_per_cluster=25, seed=7):
    """
    Four Gaussian blobs sit near the corners of a square. Diagonal corners
    share a label, exactly like the XOR truth table:
        (+1,+1) -> 0   (-1,+1) -> 1
        (+1,-1) -> 1   (-1,-1) -> 0
    No straight line can separate class 0 from class 1 — this is why a
    purely linear model (no hidden layer) is guaranteed to struggle below.
    """
    rng = random.Random(seed)
    corners = [
        ((1.0, 1.0), 0),
        ((-1.0, -1.0), 0),
        ((1.0, -1.0), 1),
        ((-1.0, 1.0), 1),
    ]
    data = []
    for (cx, cy), label in corners:
        for _ in range(points_per_cluster):
            px = cx + rng.gauss(0.0, 0.35)
            py = cy + rng.gauss(0.0, 0.35)
            data.append((px, py, label))
    rng.shuffle(data)
    return data


def train_test_split(data, test_fraction=0.25):
    n_test = int(len(data) * test_fraction)
    return data[n_test:], data[:n_test]


def evaluate_accuracy(model, dataset):
    correct = sum(1 for px, py, label in dataset if model.predict([px, py]) == label)
    return correct / len(dataset)


def train_model(model, train_data, epochs, lr, report_every):
    """Runs full-batch-per-epoch SGD (one weight update per example)."""
    history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for px, py, label in train_data:
            total_loss += model.train_step([px, py], float(label), lr)
        avg_loss = total_loss / len(train_data)
        if epoch == 1 or epoch % report_every == 0 or epoch == epochs:
            history.append((epoch, avg_loss))
    return history


# ---------------------------------------------------------------------------
# 5. ASCII DECISION BOUNDARY — the "playground" visualization
# ---------------------------------------------------------------------------

def ascii_decision_boundary(model, extent=2.2, width=45, height=21):
    """
    Sweeps a grid over [-extent, extent] on both axes and asks the model
    to classify every point. '#' marks class 1, '.' marks class 0 — the
    boundary between the two regions IS the decision boundary.
    """
    lines = []
    for row in range(height):
        y = extent - (2 * extent) * row / (height - 1)
        chars = [
            "#" if model.predict([-extent + (2 * extent) * col / (width - 1), y]) == 1
            else "."
            for col in range(width)
        ]
        lines.append("".join(chars))
    return lines


def print_loss_history(history):
    print(f"  {'Epoch':<10}Avg. Training Loss (Binary Cross-Entropy)")
    for epoch, loss in history:
        bar = "*" * min(int(loss * 40), 60)
        print(f"  {epoch:<10}{loss:.4f}  {bar}")


# ---------------------------------------------------------------------------
# DEMO RUNNER — trains one architecture and reports loss / accuracy / ASCII
# decision boundary. DEMO 1 (no hidden layer) hits a wall on XOR; DEMO 2
# (one hidden layer) and DEMO 3 (two ReLU hidden layers) both solve it.
# ---------------------------------------------------------------------------

def run_demo(demo_num, title, blurb, layer_sizes, activations, seed,
             epochs, lr, report_every, closing_note, result_name,
             train_data, test_data):
    print(THIN)
    print(f"  DEMO {demo_num}: {title}")
    print(THIN)
    print(blurb)

    model = MLP(layer_sizes=layer_sizes, activations=activations, seed=seed)
    history = train_model(model, train_data, epochs=epochs, lr=lr,
                           report_every=report_every)
    print_loss_history(history)

    train_acc = evaluate_accuracy(model, train_data)
    test_acc = evaluate_accuracy(model, test_data)
    print(f"\n  Final train accuracy: {train_acc:.1%}   Test accuracy: {test_acc:.1%}")
    print(f"  Trainable parameters: {model.param_count()}")

    print("\n  Decision boundary ('#' = predicted class 1, '.' = class 0):")
    for line in ascii_decision_boundary(model):
        print("    " + line)
    print(closing_note)

    return {
        "name": result_name,
        "params": model.param_count(),
        "final_loss": history[-1][1],
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


# ---------------------------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------------------------

def demo_comparison(results):
    print(THIN)
    print("  COMPARISON: Depth and Activation vs. Performance")
    print(THIN)

    col = (29, 11, 12, 12, 12)
    header = (
        f"  {'Architecture':<{col[0]}}{'Params':<{col[1]}}"
        f"{'Final Loss':<{col[2]}}{'Train Acc':<{col[3]}}{'Test Acc':<{col[4]}}"
    )
    print(header)
    print("  " + "-" * sum(col))
    for r in results:
        print(
            f"  {r['name']:<{col[0]}}{r['params']:<{col[1]}}"
            f"{r['final_loss']:<{col[2]}.4f}"
            f"{r['train_acc']:<{col[3]}.1%}{r['test_acc']:<{col[4]}.1%}"
        )

    print(
        "\n  What to notice:\n"
        "    - The linear model (no hidden layer) tops out far below 100%\n"
        "      accuracy — no amount of training fixes a lack of capacity.\n"
        "    - Adding just one hidden layer is enough to solve XOR, because\n"
        "      it introduces the non-linearity a straight line cannot have.\n"
        "    - The deeper ReLU network has more parameters and can converge\n"
        "      in fewer epochs, but on a dataset this small, extra depth\n"
        "      does not guarantee a better test score — it mainly buys\n"
        "      training speed and headroom for harder problems."
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print()
    print(DIVIDER)
    print("  TOPIC 03: NEURAL NETWORKS — A TINY MLP BUILT FROM SCRATCH")
    print(DIVIDER)
    print(
        "\n  This demo builds a neural network using nothing but Python lists\n"
        "  and loops: weights, biases, a forward pass, and backpropagation\n"
        "  are all implemented by hand (no numpy, no frameworks).\n"
        "\n  The task: classify points into two classes arranged in an XOR\n"
        "  pattern — a problem that is impossible to solve with a straight\n"
        "  line, which makes it a perfect showcase for why hidden layers and\n"
        "  non-linear activation functions matter.\n"
    )

    random.seed(42)  # top-level seed for overall reproducibility
    dataset = make_xor_dataset(points_per_cluster=25, seed=7)
    train_data, test_data = train_test_split(dataset, test_fraction=0.25)

    print(f"  Dataset: {len(dataset)} points total "
          f"({len(train_data)} train / {len(test_data)} test)")
    print("  Layout: 4 Gaussian clusters at the corners of a square;")
    print("  diagonal corners share a class label (the XOR pattern).\n")

    configs = [
        dict(
            demo_num=1, title="No Hidden Layer (pure logistic regression)",
            blurb=(
                "  Architecture: 2 inputs -> 1 output (sigmoid). No hidden layer\n"
                "  at all, so this model can only draw a single straight decision\n"
                "  line in the 2D plane. XOR data is NOT linearly separable, so no\n"
                "  matter how long we train, accuracy should plateau well below 100%.\n"
            ),
            layer_sizes=[2, 1], activations=["sigmoid"], seed=1,
            epochs=400, lr=0.5, report_every=100,
            closing_note=(
                "\n  As predicted: the boundary is a single straight edge, so it can\n"
                "  only ever capture two of the four XOR corners correctly."
            ),
            result_name="No hidden layer",
        ),
        dict(
            demo_num=2, title="One Hidden Layer (6 neurons, sigmoid)",
            blurb=(
                "  Architecture: 2 inputs -> 6 hidden (sigmoid) -> 1 output (sigmoid).\n"
                "  The Universal Approximation Theorem says a single hidden layer\n"
                "  with enough neurons and a non-linear activation can approximate\n"
                "  any continuous function — including XOR's checkerboard pattern.\n"
            ),
            layer_sizes=[2, 6, 1], activations=["sigmoid", "sigmoid"], seed=2,
            epochs=2500, lr=0.5, report_every=500,
            closing_note=(
                "\n  The hidden layer lets the network bend its decision surface into\n"
                "  a checkerboard-like shape, correctly isolating all four corners."
            ),
            result_name="1 hidden layer (6 sigmoid)",
        ),
        dict(
            demo_num=3, title="Two Hidden Layers (8 + 8 neurons, ReLU)",
            blurb=(
                "  Architecture: 2 inputs -> 8 hidden (ReLU) -> 8 hidden (ReLU) ->\n"
                "  1 output (sigmoid). ReLU (max(0, z)) does not saturate for\n"
                "  positive inputs, so gradients flow more easily through deeper\n"
                "  stacks — deeper nets often converge faster per epoch than a wide\n"
                "  shallow one, at the cost of more parameters to learn.\n"
            ),
            layer_sizes=[2, 8, 8, 1], activations=["relu", "relu", "sigmoid"], seed=3,
            epochs=800, lr=0.08, report_every=200,
            closing_note=(
                "\n  With more capacity, the boundary hugs the four clusters more\n"
                "  tightly — but watch the accuracy comparison below: more depth\n"
                "  isn't automatically better on a dataset this small and simple."
            ),
            result_name="2 hidden layers (8+8 ReLU)",
        ),
    ]

    results = []
    for cfg in configs:
        results.append(run_demo(train_data=train_data, test_data=test_data, **cfg))
        print()
    demo_comparison(results)

    print()
    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. A neural network is just weighted sums + activation functions,",
        "   chained across layers — nothing magical, just repeated math.",
        "2. Without a non-linear activation, stacking layers is pointless:",
        "   the whole network collapses into one linear function.",
        "3. Hidden layers give a network the capacity to bend its decision",
        "   boundary — that's precisely what solves non-linearly-separable",
        "   problems like XOR that a single linear layer cannot.",
        "4. Backpropagation is the chain rule applied layer by layer: each",
        "   layer's gradient depends only on the layer after it.",
        "5. More depth/width is not automatically better — it adds capacity",
        "   and can speed up convergence, but also adds parameters that",
        "   need enough data to be learned well (risk of overfitting).",
        "6. Everything here — training, loss, decision boundaries — was",
        "   computed with pure Python stdlib. No GPU, no libraries needed.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
