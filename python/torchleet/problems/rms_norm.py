"""rms-norm — Root Mean Square Layer Normalization from scratch."""
import torch
import torch.nn as nn

ENTRIES = ["RMSNorm"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Square the input, take the mean along the last dimension, add epsilon, then take the square root.",
    "Use keepdim=True in the mean so broadcasting works when you divide.",
    "The scale parameter gamma should have shape (dim,) and be created with nn.Parameter(torch.ones(dim)).",
]


def _ref_rms_norm(x, weight, eps=1e-8):
    """Reference implementation for checking."""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def check_is_nn_module(ns):
    assert isinstance(ns.RMSNorm(4), nn.Module), \
        "RMSNorm should be an nn.Module subclass"


def check_output_shape(ns):
    for shape in [(3, 5), (2, 4, 8), (6,)]:
        dim = shape[-1]
        m = ns.RMSNorm(dim)
        x = torch.randn(*shape)
        out = m(x)
        assert out.shape == x.shape, \
            f"output shape {tuple(out.shape)} != input shape {tuple(x.shape)} for input {shape}"


def check_has_learnable_scale(ns):
    m = ns.RMSNorm(7)
    params = list(m.parameters())
    assert len(params) == 1, \
        f"expected exactly 1 learnable parameter (scale/gamma), got {len(params)}"
    assert params[0].shape == (7,), \
        f"scale parameter shape should be (7,), got {tuple(params[0].shape)}"
    assert torch.allclose(params[0].data, torch.ones(7)), \
        "scale parameter should be initialized to ones"


def check_matches_reference(ns):
    torch.manual_seed(42)
    for shape in [(5,), (4, 8), (2, 3, 7)]:
        dim = shape[-1]
        m = ns.RMSNorm(dim)
        x = torch.randn(*shape)
        got = m(x)
        weight = list(m.parameters())[0]
        expected = _ref_rms_norm(x, weight.data, eps=m.eps)
        assert torch.allclose(got, expected, atol=1e-6), (
            f"output disagrees with reference on {shape} "
            f"(max diff {(got - expected).abs().max():.2e})")


def check_gradient_flows(ns):
    m = ns.RMSNorm(8)
    x = torch.randn(4, 8, requires_grad=True)
    loss = m(x).sum()
    loss.backward()
    scale = list(m.parameters())[0]
    assert scale.grad is not None, \
        "gradient did not flow to the scale parameter"
    assert x.grad is not None, \
        "gradient did not flow back to the input"


def check_numerically_stable(ns):
    m = ns.RMSNorm(3)
    x = torch.tensor([[1e-10, 2e-10, 3e-10]])
    out = m(x)
    assert not torch.isnan(out).any(), "NaN on very small inputs"
    assert not torch.isinf(out).any(), "Inf on very small inputs"


CHECKS = [check_is_nn_module, check_output_shape, check_has_learnable_scale,
          check_matches_reference, check_gradient_flows, check_numerically_stable]
