"""kv-cache — implement a KV cache for autoregressive generation.

The load-bearing check is self-consistency: generating token-by-token with the
cache must equal processing the whole sequence at once. Both runs use the *same*
CachedAttention instance the solver supplied, so weight initialization and op
order cannot cause a false failure — only a genuinely wrong cache can.
"""
import torch

ENTRIES = ["KVCache", "CachedAttention"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "The cache holds (batch, heads, seq_len, d_head). What changes as you decode?",
    "update() appends the new K/V along the seq_len dimension (dim=2).",
    "In cached mode you compute Q/K/V for the new token only, then attend over "
    "the full cached K/V. No causal mask is needed — the cache only holds the past.",
]

B, D_MODEL, HEADS, SEQ = 2, 64, 4, 16


def _attn(CachedAttention):
    torch.manual_seed(0)
    m = CachedAttention(D_MODEL, HEADS)
    m.eval()
    return m


def check_cache_update_shapes(ns):
    c = ns.KVCache()
    k1 = torch.randn(B, HEADS, 1, D_MODEL // HEADS)
    v1 = torch.randn(B, HEADS, 1, D_MODEL // HEADS)
    fk, fv = c.update(k1, v1)
    assert fk.shape == k1.shape, f"first update should return the new K, got {tuple(fk.shape)}"
    k2 = torch.randn(B, HEADS, 1, D_MODEL // HEADS)
    v2 = torch.randn(B, HEADS, 1, D_MODEL // HEADS)
    fk, fv = c.update(k2, v2)
    exp = (B, HEADS, 2, D_MODEL // HEADS)
    assert tuple(fk.shape) == exp, f"after 2 updates expected K {exp}, got {tuple(fk.shape)}"
    assert tuple(fv.shape) == exp, f"after 2 updates expected V {exp}, got {tuple(fv.shape)}"


def check_cache_accumulates_in_order(ns):
    c = ns.KVCache()
    ks = [torch.randn(B, HEADS, 1, D_MODEL // HEADS) for _ in range(4)]
    vs = [torch.randn(B, HEADS, 1, D_MODEL // HEADS) for _ in range(4)]
    for k, v in zip(ks, vs):
        fk, fv = c.update(k, v)
    assert torch.allclose(fk, torch.cat(ks, dim=2), atol=1e-6), \
        "cached K is not the past keys concatenated in decode order"
    assert torch.allclose(fv, torch.cat(vs, dim=2), atol=1e-6), \
        "cached V is not the past values concatenated in decode order"


def check_cached_equals_full_recomputation(ns):
    """The property that defines a correct KV cache."""
    m = _attn(ns.CachedAttention)
    x = torch.randn(B, SEQ, D_MODEL)
    with torch.no_grad():
        full = m(x, kv_cache=None)
    assert tuple(full.shape) == (B, SEQ, D_MODEL), \
        f"full forward should return {(B, SEQ, D_MODEL)}, got {tuple(full.shape)}"

    cache = ns.KVCache()
    steps = []
    with torch.no_grad():
        for t in range(SEQ):
            out_t = m(x[:, t:t + 1, :], kv_cache=cache)
            assert tuple(out_t.shape) == (B, 1, D_MODEL), \
                f"cached step {t} should return {(B, 1, D_MODEL)}, got {tuple(out_t.shape)}"
            steps.append(out_t)
    cached = torch.cat(steps, dim=1)

    diff = (full - cached).abs().max().item()
    assert torch.allclose(full, cached, atol=1e-5), (
        f"token-by-token cached generation disagrees with full recomputation "
        f"(max diff {diff:.2e}). The cache is not reproducing the causal context.")


def check_reset_clears_cache(ns):
    c = ns.KVCache()
    k = torch.randn(B, HEADS, 3, D_MODEL // HEADS)
    c.update(k, k)
    if not hasattr(c, "reset"):
        from torchleet.runner import Skip
        raise Skip("no reset() on this KVCache")
    c.reset()
    k2 = torch.randn(B, HEADS, 1, D_MODEL // HEADS)
    fk, _ = c.update(k2, k2)
    assert tuple(fk.shape) == tuple(k2.shape), \
        f"after reset() the cache should be empty, got K {tuple(fk.shape)}"


CHECKS = [
    check_cache_update_shapes,
    check_cache_accumulates_in_order,
    check_cached_equals_full_recomputation,
    check_reset_clears_cache,
]
