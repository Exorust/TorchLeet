"""custom-loss — implement Huber loss.

F.huber_loss is the oracle: it ships with PyTorch, so nothing is bundled.
"""
import torch
import torch.nn.functional as F

ENTRIES = ["HuberLoss"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Huber is quadratic for small errors and linear for large ones.",
    "The switch happens at |error| == delta.",
    "0.5*e^2 below the threshold; delta*(|e| - 0.5*delta) above it.",
]


def check_matches_torch_huber(ns):
    torch.manual_seed(0)
    for delta in (0.5, 1.0, 2.0):
        loss = ns.HuberLoss(delta=delta)
        p, t = torch.randn(64), torch.randn(64)
        got = torch.as_tensor(loss(p, t))
        exp = F.huber_loss(p, t, delta=delta)
        assert torch.allclose(got.float(), exp, atol=1e-5), \
            f"delta={delta}: got {float(got):.5f}, F.huber_loss gives {float(exp):.5f}"


def check_quadratic_below_delta(ns):
    """Small errors must behave like 0.5*e^2, not like |e|."""
    loss = ns.HuberLoss(delta=1.0)
    e = 0.2
    got = float(torch.as_tensor(loss(torch.tensor([e]), torch.tensor([0.0]))))
    assert abs(got - 0.5 * e ** 2) < 1e-6, \
        f"for |error|={e} < delta the loss should be {0.5 * e ** 2:.4f}, got {got:.4f}"


def check_linear_above_delta(ns):
    loss = ns.HuberLoss(delta=1.0)
    e = 5.0
    got = float(torch.as_tensor(loss(torch.tensor([e]), torch.tensor([0.0]))))
    want = 1.0 * (e - 0.5)
    assert abs(got - want) < 1e-6, \
        f"for |error|={e} > delta the loss should be {want:.4f}, got {got:.4f}"


def check_zero_when_perfect(ns):
    loss = ns.HuberLoss(delta=1.0)
    x = torch.randn(20)
    assert abs(float(torch.as_tensor(loss(x, x)))) < 1e-7, \
        "loss must be 0 when prediction equals target"


def check_gradients_flow(ns):
    loss = ns.HuberLoss(delta=1.0)
    p = torch.randn(16, requires_grad=True)
    torch.as_tensor(loss(p, torch.randn(16))).backward()
    assert p.grad is not None and torch.any(p.grad != 0), "no gradient w.r.t. predictions"


CHECKS = [check_matches_torch_huber, check_quadratic_below_delta,
          check_linear_above_delta, check_zero_when_perfect, check_gradients_flow]
