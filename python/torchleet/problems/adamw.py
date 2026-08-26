"""adamw — AdamW built on torch.optim.Optimizer.

MyAdamW is graded against torch's own torch.optim.AdamW, run step-for-step
from the same init on the same quadratic bowl — the trajectory must match
to float tolerance. The decoupled decay is pinned by a zero-gradient
parameter: it must still shrink by exactly (1 - lr * weight_decay). An
implementation that routes the decay through the gradient (Adam + L2) moves
it by the wrong amount, and also diverges from torch.optim.AdamW on the
trajectory check.
"""
import torch

ENTRIES = ["MyAdamW"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Subclass torch.optim.Optimizer and keep per-parameter state in "
    "self.state[p] — it starts as an empty dict, so initialize it on the "
    "first step. Do the whole update under @torch.no_grad() with in-place ops "
    "(mul_, add_, addcmul_, addcdiv_).",
    "AdamW: m = b1*m + (1-b1)*g and v = b2*v + (1-b2)*g^2, then correct both "
    "by (1 - beta**step) and update p -= lr * m_hat / (sqrt(v_hat) + eps). "
    "Weight decay is p.mul_(1 - lr * weight_decay) applied directly to the "
    "parameter — do NOT add weight_decay * p to the gradient, that is L2 "
    "regularization and the adaptive denominator rescales it.",
]


def _quadratic(d=16, seed=0):
    """A quadratic bowl f(x) = mean((x - target)^2) with a fixed init."""
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(d, generator=g)
    x0 = torch.randn(d, generator=g)
    return x0, target


def _run(opt_cls, x0, target, steps, **kw):
    """Drive an optimizer for `steps` steps on the bowl; return final params."""
    x = x0.clone().requires_grad_(True)
    opt = opt_cls([x], **kw)
    for _ in range(steps):
        loss = ((x - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return x.detach()


def check_adamw_matches_torch_builtin(ns):
    x0, target = _quadratic(seed=1)
    steps = 30
    kw = dict(lr=0.05, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05)
    mine = _run(ns.MyAdamW, x0, target, steps, **kw)
    ref = _run(torch.optim.AdamW, x0, target, steps, **kw)
    dev = float((mine - ref).abs().max())
    assert torch.allclose(mine, ref, atol=1e-5, rtol=1e-4), (
        f"after {steps} steps your AdamW is {dev:.2e} away from "
        "torch.optim.AdamW run in lockstep. AdamW = Adam plus "
        "p.mul_(1 - lr * weight_decay) applied directly to the parameter — "
        "adding weight_decay * p to the gradient instead is L2 regularization "
        "and drifts away from this trajectory")


def check_adamw_decays_params_with_zero_gradient(ns):
    """Decoupled decay works even with NO gradient signal: the moments stay
    zero, so the whole step is p *= (1 - lr * weight_decay)."""
    g = torch.Generator().manual_seed(2)
    p0 = torch.randn(8, generator=g)
    lr, wd = 0.1, 0.5
    p = p0.clone().requires_grad_(True)
    opt = ns.MyAdamW([p], lr=lr, weight_decay=wd)
    p.grad = torch.zeros_like(p)
    opt.step()
    expected = p0 * (1 - lr * wd)
    assert torch.allclose(p.detach(), expected, atol=1e-7), (
        f"a parameter with a ZERO gradient must still decay to "
        f"p * (1 - lr*weight_decay) = {float(expected[0]):.6f} on coordinate 0, "
        f"got {float(p.detach()[0]):.6f}. The decay is "
        "p.mul_(1 - lr * weight_decay) on the parameter itself — if it goes "
        "through the gradient, a zero gradient means no decay (or a decay "
        "warped by the adaptive denominator)")


CHECKS = [
    check_adamw_matches_torch_builtin,
    check_adamw_decays_params_with_zero_gradient,
]
