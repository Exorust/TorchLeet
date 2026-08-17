"""multi-head-attention.

The reference builds its projections inside the function, so weights are random
per call. Checks therefore seed before each call rather than comparing outputs
across calls.
"""
import torch

ENTRIES = ["multi_head_attention"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Project Q/K/V, split d_model into num_heads chunks of d_head, attend, then concatenate.",
    "Reshape to (batch, heads, seq, d_head) with view + transpose(1, 2).",
    "After attention, transpose back and merge the heads before the output projection.",
]

B, S, D_MODEL, HEADS = 2, 7, 32, 4


def _in(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, S, D_MODEL)
    return x, x.clone(), x.clone()


def _out(r):
    return r[0] if isinstance(r, tuple) else r


def check_output_shape(ns):
    q, k, v = _in()
    out = _out(ns.multi_head_attention(q, k, v, HEADS, D_MODEL))
    assert tuple(out.shape) == (B, S, D_MODEL), \
        f"expected {(B, S, D_MODEL)}, got {tuple(out.shape)}"


def check_deterministic_under_seed(ns):
    q, k, v = _in()
    torch.manual_seed(123)
    a = _out(ns.multi_head_attention(q, k, v, HEADS, D_MODEL))
    torch.manual_seed(123)
    b = _out(ns.multi_head_attention(q, k, v, HEADS, D_MODEL))
    assert torch.allclose(a, b, atol=1e-6), \
        "same seed and same input gave different outputs"


def check_head_count_changes_result(ns):
    """Splitting into heads must actually change the computation."""
    q, k, v = _in(3)
    torch.manual_seed(7)
    one = _out(ns.multi_head_attention(q, k, v, 1, D_MODEL))
    torch.manual_seed(7)
    many = _out(ns.multi_head_attention(q, k, v, 8, D_MODEL))
    assert not torch.allclose(one, many, atol=1e-5), \
        "1 head and 8 heads produced identical output — heads are not being split"


def check_mask_changes_result(ns):
    q, k, v = _in(4)
    mask = torch.ones(S, S, dtype=torch.bool).tril()
    torch.manual_seed(11)
    plain = _out(ns.multi_head_attention(q, k, v, HEADS, D_MODEL))
    torch.manual_seed(11)
    masked = _out(ns.multi_head_attention(q, k, v, HEADS, D_MODEL, mask=mask))
    assert not torch.allclose(plain, masked, atol=1e-5), \
        "passing a causal mask changed nothing — the mask is ignored"


CHECKS = [check_output_shape, check_deterministic_under_seed,
          check_head_count_changes_result, check_mask_changes_result]
