"""logistic-regression — sigmoid, BCE loss, and a training loop."""
import torch
import torch.nn.functional as F

ENTRIES = ["sigmoid", "bce_loss", "train_logistic_regression"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "sigmoid(z) = 1 / (1 + exp(-z)).",
    "BCE = -mean(y*log(p) + (1-y)*log(1-p)); clamp p away from 0 and 1 first.",
    "Train with gradient descent on the BCE of sigmoid(Xw + b).",
]


def check_sigmoid_matches_torch(ns):
    z = torch.linspace(-8, 8, 40)
    got = ns.sigmoid(z)
    assert torch.allclose(got, torch.sigmoid(z), atol=1e-6), \
        f"disagrees with torch.sigmoid (max diff {(got - torch.sigmoid(z)).abs().max():.2e})"


def check_sigmoid_range_and_midpoint(ns):
    out = ns.sigmoid(torch.linspace(-50, 50, 101))
    assert bool(((out >= 0) & (out <= 1)).all()), "sigmoid must stay within [0, 1]"
    assert abs(float(ns.sigmoid(torch.zeros(1))) - 0.5) < 1e-6, "sigmoid(0) must be 0.5"


def check_bce_matches_torch(ns):
    torch.manual_seed(0)
    p = torch.rand(50).clamp(0.02, 0.98)
    y = (torch.rand(50) > 0.5).float()
    got = ns.bce_loss(y, p)
    exp = F.binary_cross_entropy(p, y)
    assert torch.allclose(torch.as_tensor(got).float(), exp, atol=1e-4), \
        f"disagrees with F.binary_cross_entropy ({float(got):.5f} vs {float(exp):.5f})"


def check_bce_is_lower_when_confident_and_right(ns):
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    good = ns.bce_loss(y, torch.tensor([0.95, 0.93, 0.05, 0.02]))
    bad = ns.bce_loss(y, torch.tensor([0.10, 0.20, 0.85, 0.90]))
    assert float(good) < float(bad), \
        "confident correct predictions must score a lower loss than confident wrong ones"


def check_training_separates_two_classes(ns):
    torch.manual_seed(0)
    X = torch.cat([torch.randn(60, 2) - 2.5, torch.randn(60, 2) + 2.5])
    y = torch.cat([torch.zeros(60), torch.ones(60)])
    out = ns.train_logistic_regression(X, y)
    w, b = (out + (None,))[:2] if isinstance(out, tuple) else (out, None)
    assert w is not None, "train_logistic_regression returned nothing usable"
    logits = X @ torch.as_tensor(w).reshape(-1) + (float(b) if b is not None else 0.0)
    acc = ((torch.sigmoid(logits) > 0.5).float() == y).float().mean()
    assert float(acc) > 0.9, \
        f"only {float(acc):.0%} accuracy on linearly separable data — training is not converging"


CHECKS = [check_sigmoid_matches_torch, check_sigmoid_range_and_midpoint,
          check_bce_matches_torch, check_bce_is_lower_when_confident_and_right,
          check_training_separates_two_classes]
