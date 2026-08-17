"""mixture-of-experts — top-k routed MoE layer with load balancing."""
from types import SimpleNamespace

import torch

ENTRIES = ["Expert", "MoELayer"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Route each token to its top-k experts and weight their outputs by the gate probabilities.",
    "Renormalize the top-k gate values so each token's weights sum to 1.",
    "The aux loss penalizes imbalance: experts should receive roughly equal token shares.",
]

B, S, D_MODEL, D_FF, N_EXP, TOP_K = 2, 8, 64, 128, 8, 2


def _cfg(**kw):
    base = dict(num_experts=N_EXP, top_k=TOP_K, d_model=D_MODEL, d_ff=D_FF,
                aux_loss_weight=0.01)
    base.update(kw)
    return SimpleNamespace(**base)


def _layer(ns, **kw):
    torch.manual_seed(0)
    return ns.MoELayer(_cfg(**kw))


def check_expert_shape(ns):
    e = ns.Expert(D_MODEL, D_FF)
    out = e(torch.randn(5, D_MODEL))
    assert tuple(out.shape) == (5, D_MODEL), \
        f"an expert must preserve d_model: expected {(5, D_MODEL)}, got {tuple(out.shape)}"


def check_layer_output_shape(ns):
    out = _layer(ns)(torch.randn(B, S, D_MODEL))
    out = out[0] if isinstance(out, tuple) else out
    assert tuple(out.shape) == (B, S, D_MODEL), \
        f"expected {(B, S, D_MODEL)}, got {tuple(out.shape)}"


def check_returns_aux_loss(ns):
    res = _layer(ns)(torch.randn(B, S, D_MODEL))
    assert isinstance(res, tuple) and len(res) == 2, \
        "MoELayer.forward should return (output, aux_loss)"
    aux = torch.as_tensor(res[1])
    assert aux.ndim == 0, f"aux_loss should be a scalar, got shape {tuple(aux.shape)}"
    assert float(aux) >= 0, f"aux_loss should be non-negative, got {float(aux):.4f}"


def check_creates_one_module_per_expert(ns):
    layer = _layer(ns, num_experts=5)
    found = sum(1 for m in layer.modules() if isinstance(m, ns.Expert))
    assert found == 5, f"config asked for 5 experts, layer built {found}"


def check_gradients_flow(ns):
    layer = _layer(ns)
    x = torch.randn(B, S, D_MODEL, requires_grad=True)
    res = layer(x)
    (res[0] if isinstance(res, tuple) else res).sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0), \
        "no gradient reached the input — the routing path is detached"


def check_routing_is_input_dependent(ns):
    """A router that ignores its input is not routing."""
    layer = _layer(ns)
    a = layer(torch.randn(B, S, D_MODEL) * 5)
    b = layer(torch.randn(B, S, D_MODEL) * 5)
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    assert not torch.allclose(a, b, atol=1e-4), \
        "different inputs produced identical outputs"


CHECKS = [check_expert_shape, check_layer_output_shape, check_returns_aux_loss,
          check_creates_one_module_per_expert, check_gradients_flow,
          check_routing_is_input_dependent]
