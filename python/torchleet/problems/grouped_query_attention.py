"""grouped-query-attention — GQA shares K/V heads across groups of query heads.

The reference builds its projections inside the function, so weights are random
per call; checks seed before each call rather than comparing across calls.
"""
import torch

ENTRIES = ["grouped_query_attention"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Queries keep all their heads; keys and values only have num_query_groups heads.",
    "Project K/V to num_query_groups * d_head, not to d_model — that is where the saving comes from.",
    "repeat_interleave each K/V head across the query heads that share it, then attend as usual.",
]

B, S, D_MODEL, HEADS = 2, 6, 32, 4


def _out(r):
    return r[0] if isinstance(r, tuple) else r


def _qkv(seed=0, d_model=D_MODEL):
    torch.manual_seed(seed)
    x = torch.randn(B, S, d_model)
    return x, x.clone(), x.clone()


def _call(ns, groups, seed=0, mask=None, heads=HEADS, d_model=D_MODEL):
    q, k, v = _qkv(seed, d_model)
    return _out(ns.grouped_query_attention(
        q, k, v, num_query_heads=heads, num_query_groups=groups,
        d_model=d_model, mask=mask))


def check_output_shape(ns):
    for groups in (1, 2, 4):
        torch.manual_seed(3)
        out = _call(ns, groups)
        assert tuple(out.shape) == (B, S, D_MODEL), \
            f"num_query_groups={groups}: expected {(B, S, D_MODEL)}, got {tuple(out.shape)}"


def check_groups_must_divide_heads(ns):
    """3 K/V groups cannot evenly serve 4 query heads."""
    try:
        _call(ns, 3)
    except Exception:
        return
    raise AssertionError(
        "num_query_groups=3 with num_query_heads=4 should be rejected — "
        "every K/V group must serve the same number of query heads")


def check_deterministic_under_seed(ns):
    torch.manual_seed(5)
    a = _call(ns, 2, seed=1)
    torch.manual_seed(5)
    b = _call(ns, 2, seed=1)
    assert torch.allclose(a, b, atol=1e-6), \
        "same seed and same input gave different outputs"


def check_output_is_finite(ns):
    out = _call(ns, 2, seed=4)
    assert not torch.isnan(out).any(), "NaN in the output"
    assert not torch.isinf(out).any(), "Inf in the output"


def check_causal_mask_hides_the_future(ns):
    """With a causal mask, changing the last position must not move earlier rows."""
    q, k, v = _qkv(0)
    causal = torch.tril(torch.ones(S, S)).bool()

    torch.manual_seed(21)
    base = _out(ns.grouped_query_attention(
        q, k, v, num_query_heads=HEADS, num_query_groups=2,
        d_model=D_MODEL, mask=causal))

    k2, v2 = k.clone(), v.clone()
    k2[:, -1, :] += 5.0
    v2[:, -1, :] += 5.0
    torch.manual_seed(21)
    perturbed = _out(ns.grouped_query_attention(
        q, k2, v2, num_query_heads=HEADS, num_query_groups=2,
        d_model=D_MODEL, mask=causal))

    assert torch.allclose(base[:, :-1, :], perturbed[:, :-1, :], atol=1e-5), (
        "rewriting the final key/value changed earlier outputs — "
        "the causal mask is not being applied")


def check_gradients_flow(ns):
    torch.manual_seed(0)
    q = torch.randn(B, S, D_MODEL, requires_grad=True)
    k, v = torch.randn(B, S, D_MODEL), torch.randn(B, S, D_MODEL)
    _out(ns.grouped_query_attention(
        q, k, v, num_query_heads=HEADS, num_query_groups=2,
        d_model=D_MODEL)).sum().backward()
    assert q.grad is not None and torch.any(q.grad != 0), \
        "no gradient reached the queries"


CHECKS = [check_output_shape, check_groups_must_divide_heads,
          check_deterministic_under_seed, check_output_is_finite,
          check_causal_mask_hides_the_future, check_gradients_flow]
