"""custom-autograd — Learned-SiLU as a torch.autograd.Function.

Two oracles do the work and neither ships an implementation. The forward pass is
compared against `slope * F.silu(x)`, which is just the formula in the problem
statement expressed with a torch built-in. The backward pass is checked with
`torch.autograd.gradcheck`, which compares the hand-written gradients against
finite differences of the solver's *own* forward — so any correct backward
passes, however it is written, and an incorrect one cannot.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["LearnedSiLUFunction", "LinearRegressionModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "forward(ctx, x, slope) returns slope * x * sigmoid(x); ctx.save_for_backward "
    "whatever the backward pass needs.",
    "d/dx [x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1 - sigmoid(x)); multiply by "
    "slope and by grad_output.",
    "backward must return one gradient per forward input: d/dslope is "
    "grad_output * x * sigmoid(x). Keep the slope an nn.Parameter in the model so "
    "the optimizer updates it.",
]


def _apply(ns, x, slope):
    """Call the solver's function the way autograd.Function is meant to be used."""
    fn = ns.LearnedSiLUFunction
    if hasattr(fn, "apply"):
        return fn.apply(x, slope)
    return fn()(x, slope)          # tolerate the nn.Module variant


def check_forward_matches_the_formula(ns):
    torch.manual_seed(0)
    x = torch.linspace(-8, 8, 128)
    for s in (0.5, 1.0, 2.5):
        slope = torch.tensor(s)
        got = _apply(ns, x, slope)
        exp = s * F.silu(x)
        assert torch.allclose(got, exp, atol=1e-5), (
            f"slope={s}: output disagrees with slope * x * sigmoid(x) "
            f"(max diff {(got - exp).abs().max():.2e})")


def check_is_an_autograd_function(ns):
    assert isinstance(ns.LearnedSiLUFunction, type) and \
        issubclass(ns.LearnedSiLUFunction, torch.autograd.Function), (
            "LearnedSiLUFunction should subclass torch.autograd.Function so you "
            "write forward and backward yourself")


def check_backward_matches_finite_differences(ns):
    """gradcheck differentiates the solver's own forward numerically."""
    torch.manual_seed(0)
    x = torch.randn(12, dtype=torch.double, requires_grad=True)
    slope = torch.tensor([1.7], dtype=torch.double, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda a, b: _apply(ns, a, b), (x, slope),
                                  eps=1e-6, atol=1e-4, raise_exception=False)
    assert ok, (
        "the hand-written backward does not match a numerical derivative of your "
        "own forward — check both returned gradients: d/dx is "
        "grad_output * slope * (sig + x*sig*(1-sig)) and d/dslope is "
        "grad_output * x * sig")


def check_gradient_reaches_the_input(ns):
    x = torch.randn(16, requires_grad=True)
    _apply(ns, x, torch.tensor(1.0)).sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0), \
        "no gradient reached x — backward returned None for the first input"


def check_model_applies_the_activation(ns):
    """The model's ctor arg is the initial slope, so its output is pinned."""
    for s in (1.0, 3.0):
        torch.manual_seed(0)
        m = ns.LinearRegressionModel(s)
        x = torch.randn(10, 1)
        got = m(x)
        exp = s * F.silu(x)
        assert tuple(got.shape) == tuple(x.shape), (
            f"the activation is elementwise: expected {tuple(x.shape)} out for "
            f"{tuple(x.shape)} in, got {tuple(got.shape)}")
        assert torch.allclose(got, exp, atol=1e-5), (
            f"LinearRegressionModel({s}) should start out computing "
            f"{s} * x * sigmoid(x) (max diff {(got - exp).abs().max():.2e})")


def check_slope_is_learnable(ns):
    torch.manual_seed(0)
    m = ns.LinearRegressionModel(1.0)
    params = [p for p in m.parameters() if p.requires_grad]
    assert params, (
        "the model has no trainable parameters — the slope must be an "
        "nn.Parameter, not a plain float")
    x = torch.randn(32, 1)
    target = 4.0 * F.silu(x)
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    lossf = nn.MSELoss()
    first = lossf(m(x), target).item()
    loss = None
    for _ in range(100):
        opt.zero_grad()
        loss = lossf(m(x), target)
        loss.backward()
        opt.step()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in params), \
        "no gradient reached the slope parameter — is it returned by backward?"
    assert loss.item() < first, (
        f"training could not fit a target that is just a rescaled Learned-SiLU "
        f"({first:.4f} -> {loss.item():.4f}) — the slope is not being updated")


CHECKS = [
    check_is_an_autograd_function,
    check_forward_matches_the_formula,
    check_backward_matches_finite_differences,
    check_gradient_reaches_the_input,
    check_model_applies_the_activation,
    check_slope_is_learnable,
]
