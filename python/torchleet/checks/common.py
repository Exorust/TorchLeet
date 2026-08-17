"""Reusable check factories.

Most `torch/` problems are "build a module that maps this shape to that shape and
trains". Rather than hand-write four near-identical checks per problem, specs
compose these factories. Each returns a named function so failure output still
reads as a specific check.

Everything here verifies *properties*, never stored outputs — a correct model with
a different architecture must pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _named(fn, name):
    fn.__name__ = name
    return fn


def build(ns, entry, *args, seed=0, **kwargs):
    """Instantiate the solver's class with a fixed seed."""
    torch.manual_seed(seed)
    return getattr(ns, entry)(*args, **kwargs)


def output_shape(entry, in_shape, out_shape, ctor=(), batch=4):
    """Model maps (batch, *in_shape) -> (batch, *out_shape)."""
    def check_output_shape(ns):
        m = build(ns, entry, *ctor)
        out = m(torch.randn(batch, *in_shape))
        want = (batch, *out_shape)
        assert tuple(out.shape) == want, \
            f"expected output {want} for input {(batch, *in_shape)}, got {tuple(out.shape)}"
    return check_output_shape


def batch_agnostic(entry, in_shape, out_shape, ctor=(), sizes=(1, 3, 8)):
    def check_batch_agnostic(ns):
        m = build(ns, entry, *ctor)
        for b in sizes:
            out = m(torch.randn(b, *in_shape))
            assert tuple(out.shape) == (b, *out_shape), \
                f"batch {b} gave {tuple(out.shape)}, expected {(b, *out_shape)}"
    return check_batch_agnostic


def gradients_flow(entry, in_shape, ctor=(), batch=2):
    """Every trainable parameter must receive a gradient."""
    def check_gradients_reach_all_parameters(ns):
        m = build(ns, entry, *ctor)
        if not list(m.parameters()):
            from torchleet.runner import Skip
            raise Skip("module has no parameters")
        m(torch.randn(batch, *in_shape)).sum().backward()
        dead = [n for n, p in m.named_parameters()
                if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
        assert not dead, (
            f"no gradient reached: {', '.join(dead[:4])}"
            f"{'...' if len(dead) > 4 else ''} — not on the forward path")
    return check_gradients_reach_all_parameters


def can_learn(entry, in_shape, out_shape, ctor=(), batch=8, steps=40, lr=1e-3,
              classification=False):
    """Overfit one fixed batch; the loss must fall. Catches a model that is wired
    together but cannot train."""
    def check_can_learn(ns):
        torch.manual_seed(0)
        m = build(ns, entry, *ctor)
        if not list(m.parameters()):
            from torchleet.runner import Skip
            raise Skip("module has no parameters")
        x = torch.randn(batch, *in_shape)
        if classification:
            y = torch.randint(0, out_shape[-1], (batch,))
            lossf = nn.CrossEntropyLoss()
        else:
            y = torch.randn(batch, *out_shape)
            lossf = nn.MSELoss()
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        first = lossf(m(x), y).item()
        loss = None
        for _ in range(steps):
            opt.zero_grad()
            loss = lossf(m(x), y)
            loss.backward()
            opt.step()
        assert loss.item() < first, (
            f"loss did not decrease over {steps} steps on a fixed batch "
            f"({first:.4f} -> {loss.item():.4f}) — the model is not learning")
    return check_can_learn


def deterministic(entry, in_shape, ctor=(), batch=2):
    """Same input, same weights, eval mode -> same output."""
    def check_deterministic_in_eval(ns):
        m = build(ns, entry, *ctor)
        m.eval()
        x = torch.randn(batch, *in_shape)
        with torch.no_grad():
            a, b = m(x), m(x)
        assert torch.allclose(a, b, atol=1e-6), \
            "two forward passes on the same input disagree in eval mode"
    return check_deterministic_in_eval


def elementwise_matches(entry, oracle, name, lo=-6.0, hi=6.0, n=100, atol=1e-6):
    """Compare a pure elementwise function against a torch built-in."""
    def check(ns):
        x = torch.linspace(lo, hi, n)
        got = getattr(ns, entry)(x)
        exp = oracle(x)
        assert torch.allclose(got, exp, atol=atol), \
            f"disagrees with {name} (max diff {(got - exp).abs().max():.2e})"
    return _named(check, f"check_matches_{name}")


def preserves_shape(entry, shape=(4, 16)):
    def check_preserves_shape(ns):
        out = getattr(ns, entry)(torch.randn(*shape))
        assert tuple(out.shape) == shape, \
            f"expected the input shape {shape} back, got {tuple(out.shape)}"
    return check_preserves_shape


def no_nan(entry, shape=(4, 16), scale=1e4):
    def check_stable_on_large_inputs(ns):
        out = getattr(ns, entry)(torch.randn(*shape) * scale)
        assert not torch.isnan(out).any(), "NaN on large inputs"
        assert not torch.isinf(out).any(), "Inf on large inputs"
    return check_stable_on_large_inputs
