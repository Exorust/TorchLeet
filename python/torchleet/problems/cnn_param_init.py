"""cnn-param-init — a factory of weight-initialisation strategies.

`config_init(name)` returns something you can hand to `Module.apply`, so the
checks apply it to a throwaway CNN and measure what came out. Nothing is compared
against a reference implementation: each strategy is defined by the *distribution*
it produces, and the layers used here are square (fan_in == fan_out == n) so the
uniform and normal flavours of Xavier and Kaiming have the same standard
deviation — sqrt(1/n) and sqrt(2/n) respectively, exactly as the problem states.
"""
import math

import torch
import torch.nn as nn

ENTRIES = ["config_init"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "config_init returns the initializer *function*, it does not run it — you use "
    "it as `model.apply(config_init('xavier'))`.",
    "Each initializer takes one module and should only touch it when it is an "
    "nn.Conv2d or nn.Linear; apply() walks containers and activations too.",
    "torch.nn.init has all four: zeros_, normal_, xavier_normal_ (variance 1/n) "
    "and kaiming_normal_ (variance 2/n). Do not forget the bias.",
]

STRATEGIES = ("zeros", "random", "xavier", "kaiming")
CONV_FAN = 32 * 3 * 3        # fan_in == fan_out for a 32->32 3x3 conv
FC_FAN = 32 * 4 * 4          # ...and for the Linear that follows the pool


class _Probe(nn.Module):
    """Square layers, plus modules an initializer must learn to ignore."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(FC_FAN, FC_FAN)

    def forward(self, x):
        return self.fc(self.pool(self.relu(self.conv(x))).flatten(1))


def _initialised(ns, strategy):
    torch.manual_seed(0)
    model = _Probe()
    init = ns.config_init(strategy)
    assert init is not None, (
        f"config_init({strategy!r}) returned None — every strategy in "
        f"{list(STRATEGIES)} needs an entry in the dict")
    assert callable(init), (
        f"config_init({strategy!r}) returned a {type(init).__name__}; it should "
        "return a function that Module.apply can call on each submodule")
    model.apply(init)
    return model


def check_every_strategy_returns_an_initializer(ns):
    for strategy in STRATEGIES:
        init = ns.config_init(strategy)
        assert init is not None and callable(init), (
            f"config_init({strategy!r}) gave {init!r}; each of "
            f"{list(STRATEGIES)} must map to a callable initializer")


def check_survives_apply_on_a_whole_model(ns):
    """apply() visits ReLU, MaxPool and the container itself, not just the layers."""
    for strategy in STRATEGIES:
        model = _initialised(ns, strategy)
        out = model(torch.randn(2, 32, 8, 8))
        assert tuple(out.shape) == (2, FC_FAN), (
            f"after {strategy!r} initialisation the model returned "
            f"{tuple(out.shape)} — an initializer must not change any shapes")


def check_zeros_sets_everything_to_zero(ns):
    model = _initialised(ns, "zeros")
    for name, p in model.named_parameters():
        assert torch.all(p == 0), (
            f"{name} is not all zero after the zeros strategy "
            f"(max |value| {float(p.abs().max()):.4f}) — weights *and* biases go "
            "to zero, for both Conv2d and Linear")


def check_random_is_a_unit_normal(ns):
    model = _initialised(ns, "random")
    for name, w in (("conv.weight", model.conv.weight), ("fc.weight", model.fc.weight)):
        std = float(w.detach().std())
        assert 0.75 < std < 1.35, (
            f"{name} has std {std:.4f} after the random strategy; the problem asks "
            "for a plain standard normal (nn.init.normal_ defaults, std 1.0)")


def _expect_std(ns, strategy, factor):
    model = _initialised(ns, strategy)
    for name, w, fan in (("conv.weight", model.conv.weight, CONV_FAN),
                         ("fc.weight", model.fc.weight, FC_FAN)):
        want = math.sqrt(factor / fan)
        std = float(w.detach().std())
        assert 0.75 * want < std < 1.3 * want, (
            f"{name} has std {std:.4f} after the {strategy!r} strategy, but "
            f"variance {factor}/n with n={fan} means std {want:.4f} — "
            f"off by {std / want:.2f}x")


def check_xavier_scales_by_one_over_n(ns):
    """Xavier: variance 1/n, i.e. 2/(fan_in + fan_out)."""
    _expect_std(ns, "xavier", 1.0)


def check_kaiming_scales_by_two_over_n(ns):
    """Kaiming He: variance 2/n — twice Xavier, to survive ReLU."""
    _expect_std(ns, "kaiming", 2.0)


def check_strategies_actually_differ(ns):
    """Four names must not quietly be the same initializer."""
    stds = {}
    for strategy in STRATEGIES:
        model = _initialised(ns, strategy)
        stds[strategy] = float(model.conv.weight.detach().std())
    assert len(set(round(v, 4) for v in stds.values())) == len(STRATEGIES), (
        f"different strategies produced the same weight spread: {stds} — each "
        "name should map to its own initializer")


CHECKS = [
    check_every_strategy_returns_an_initializer,
    check_survives_apply_on_a_whole_model,
    check_zeros_sets_everything_to_zero,
    check_random_is_a_unit_normal,
    check_xavier_scales_by_one_over_n,
    check_kaiming_scales_by_two_over_n,
    check_strategies_actually_differ,
]
