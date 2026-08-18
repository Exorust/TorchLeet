"""cnn-scratch — convolution and max-pool implemented by hand.

nn.Conv2d / F.max_pool2d are the oracles. The solver's own weights are copied
into the reference layer, so only the convolution arithmetic is compared.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["Conv2dCustom", "MaxPool2dCustom"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Output size is (H + 2*padding - kernel) // stride + 1 per spatial dim.",
    "F.unfold turns sliding windows into columns if you want to avoid Python loops.",
    "Do not forget to add the bias, once per output channel.",
]


def check_conv_output_shape(ns):
    for stride, pad in ((1, 0), (1, 1), (2, 1)):
        m = ns.Conv2dCustom(3, 8, 3, stride=stride, padding=pad)
        out = m(torch.randn(2, 3, 16, 16))
        exp = (16 + 2 * pad - 3) // stride + 1
        assert tuple(out.shape) == (2, 8, exp, exp), \
            f"stride={stride} padding={pad}: expected {(2, 8, exp, exp)}, got {tuple(out.shape)}"


def check_conv_matches_torch(ns):
    """Same weights in, same numbers out."""
    torch.manual_seed(0)
    for stride, pad in ((1, 0), (2, 1)):
        m = ns.Conv2dCustom(3, 4, 3, stride=stride, padding=pad)
        ref = nn.Conv2d(3, 4, 3, stride=stride, padding=pad)
        with torch.no_grad():
            ref.weight.copy_(m.weight)
            ref.bias.copy_(m.bias)
        x = torch.randn(2, 3, 12, 12)
        got, exp = m(x), ref(x)
        assert torch.allclose(got, exp, atol=1e-5), (
            f"stride={stride} padding={pad}: disagrees with nn.Conv2d "
            f"(max diff {(got - exp).abs().max():.2e})")


def check_conv_gradients_flow(ns):
    m = ns.Conv2dCustom(3, 4, 3)
    m(torch.randn(1, 3, 10, 10)).sum().backward()
    assert m.weight.grad is not None and torch.any(m.weight.grad != 0), \
        "no gradient reached the conv weights"


def check_maxpool_matches_torch(ns):
    torch.manual_seed(0)
    p = ns.MaxPool2dCustom(2, stride=2)
    x = torch.randn(2, 3, 8, 8)
    got, exp = p(x), F.max_pool2d(x, 2, stride=2)
    assert tuple(got.shape) == tuple(exp.shape), \
        f"expected {tuple(exp.shape)}, got {tuple(got.shape)}"
    assert torch.allclose(got, exp, atol=1e-6), \
        f"disagrees with F.max_pool2d (max diff {(got - exp).abs().max():.2e})"


CHECKS = [check_conv_output_shape, check_conv_matches_torch,
          check_conv_gradients_flow, check_maxpool_matches_torch]
