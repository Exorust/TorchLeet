"""adam — Adam built on torch.optim.Optimizer.

MyAdam is graded against torch's own torch.optim.Adam, run step-for-step
from the same init on the same quadratic bowl. The built-in is the oracle,
so the trajectory must match to float tolerance. The bias correction is
pinned by a hand-derived first step: from zero state, one Adam step moves
every coordinate by almost exactly lr * sign(grad). Skipping the
(1 - beta**step) corrections overshoots by ~3.16x, a plain SGD update moves
by lr * grad instead — both fail loudly.
"""
import torch

ENTRIES = ["MyAdam"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Subclass torch.optim.Optimizer and keep per-parameter state in "
    "self.state[p] — it starts as an empty dict, so initialize it on the "
    "first step. Do the whole update under @torch.no_grad() with in-place ops "
    "(mul_, add_, addcmul_, addcdiv_).",
    "Adam: m = b1*m + (1-b1)*g and v = b2*v + (1-b2)*g^2, then correct both "
    "by (1 - beta**step) and update p -= lr * m_hat / (sqrt(v_hat) + eps).",
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


def check_adam_matches_torch_builtin(ns):
    x0, target = _quadratic()
    steps = 30
    kw = dict(lr=0.05, betas=(0.9, 0.99), eps=1e-8)
    mine = _run(ns.MyAdam, x0, target, steps, **kw)
    ref = _run(torch.optim.Adam, x0, target, steps, **kw)
    dev = float((mine - ref).abs().max())
    assert torch.allclose(mine, ref, atol=1e-5, rtol=1e-4), (
        f"after {steps} steps on the same quadratic your Adam is {dev:.2e} away "
        "from torch.optim.Adam run in lockstep. The update must be "
        "p -= lr * (m / (1 - b1**t)) / (sqrt(v / (1 - b2**t)) + eps) with "
        "m = b1*m + (1-b1)*g and v = b2*v + (1-b2)*g^2 — check the moment "
        "updates, the bias corrections, and where eps lands")


def check_adam_first_step_is_lr_times_sign_grad(ns):
    """Hand-derived: from zero state, m_hat = g and v_hat = g^2, so one step
    moves each coordinate by lr in the -sign(grad) direction. This pins the
    bias correction with no oracle at all."""
    g = torch.Generator().manual_seed(1)
    p0 = torch.randn(8, generator=g)
    grad = torch.randn(8, generator=g) * 10.0  # |g| >> eps so eps is negligible
    p = p0.clone().requires_grad_(True)
    opt = ns.MyAdam([p], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    p.grad = grad.clone()
    opt.step()
    delta = p.detach() - p0
    expected = -0.01 * grad.sign()
    assert torch.allclose(delta, expected, atol=1e-6), (
        f"from zero state one Adam step should be -lr*sign(grad) = "
        f"{expected[0]:.6f} on coordinate 0, got {float(delta[0]):.6f}. "
        "If you are ~3.2x too large you skipped the (1 - beta**step) bias "
        "corrections; if you moved by lr*grad you wrote SGD; if nothing moved "
        "the update never reached the parameter")


def check_adam_reduces_loss_on_quadratic(ns):
    x0, target = _quadratic(seed=3)
    first = float(((x0 - target) ** 2).mean())
    x = x0.clone().requires_grad_(True)
    opt = ns.MyAdam([x], lr=0.1)
    for _ in range(200):
        loss = ((x - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = float(((x - target) ** 2).mean())
    assert final < first * 0.01, (
        f"200 Adam steps on a quadratic bowl should nearly solve it: loss went "
        f"{first:.4f} -> {final:.4f}. If it diverged, the sign of the update or "
        "the eps placement is wrong; if it barely moved, check the bias "
        "correction and the lr")


CHECKS = [
    check_adam_matches_torch_builtin,
    check_adam_first_step_is_lr_times_sign_grad,
    check_adam_reduces_loss_on_quadratic,
]
