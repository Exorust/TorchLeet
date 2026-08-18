"""smollm — the RoPE + grouped-query attention block a small Llama-style LM is built from.

The manifest lists `transformers` as an extra because the notebook loads a real
checkpoint to compare generations. None of the graded entries need it: they are
plain torch, so they are checked everywhere rather than skipped.
"""
import math
from types import SimpleNamespace

import torch

ENTRIES = ["rotate_half", "apply_rotary_pos_emb", "repeat_kv",
           "RotaryEmbedder", "RopeAttention"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "rotate_half splits the last dim in half and returns (-x2, x1) concatenated.",
    "repeat_kv expands each KV head n_rep times *in place* so head i of Q pairs "
    "with KV head i // n_rep — expand + reshape, not a plain repeat/tile.",
    "RopeAttention: project, split into num_heads (Q) and kv_heads (K/V), rotate Q and K, "
    "repeat_kv the K/V, then scaled dot-product attention plus the additive mask.",
]

B, HEADS, KV_HEADS, S, HEAD_DIM = 2, 4, 2, 6, 8
HIDDEN = HEADS * HEAD_DIM


def _config():
    return SimpleNamespace(hidden_size=HIDDEN, num_heads=HEADS, kv_heads=KV_HEADS,
                           num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
                           intermediate_size=4 * HIDDEN, rope_theta=10000.0)


def _attn(ns, seed=0):
    torch.manual_seed(seed)
    return ns.RopeAttention(_config()).eval()


def _causal(seq=S):
    """Additive mask, the shape RopeAttention's `attn_weights + attention_mask` wants."""
    m = torch.zeros(1, 1, seq, seq)
    m.masked_fill_(torch.ones(seq, seq, dtype=torch.bool).triu(1), float("-inf"))
    return m


def check_rotate_half_negates_when_applied_twice(ns):
    x = torch.randn(B, HEADS, S, HEAD_DIM)
    once = ns.rotate_half(x)
    assert tuple(once.shape) == tuple(x.shape), \
        f"rotate_half must preserve shape {tuple(x.shape)}, got {tuple(once.shape)}"
    assert torch.allclose(ns.rotate_half(once), -x, atol=1e-6), \
        "applying rotate_half twice should give exactly -x (it is a 90 degree rotation)"


def check_rotary_pos_emb_identity(ns):
    """cos=1, sin=0 is a zero-angle rotation, so q and k must come back untouched."""
    q, k = torch.randn(B, HEADS, S, HEAD_DIM), torch.randn(B, KV_HEADS, S, HEAD_DIM)
    cos, sin = torch.ones(1, S, HEAD_DIM), torch.zeros(1, S, HEAD_DIM)
    rq, rk = ns.apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(rq, q, atol=1e-6), "cos=1, sin=0 must leave q unchanged"
    assert torch.allclose(rk, k, atol=1e-6), "cos=1, sin=0 must leave k unchanged"


def check_rotary_pos_emb_preserves_norm(ns):
    """RoPE is a rotation — it can move a vector but never stretch it."""
    torch.manual_seed(0)
    q, k = torch.randn(B, HEADS, S, HEAD_DIM), torch.randn(B, KV_HEADS, S, HEAD_DIM)
    ang = torch.rand(1, S, HEAD_DIM // 2) * 6.28
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    rq, _ = ns.apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-4), (
        "rotating changed the vector norms (max change "
        f"{float((rq.norm(dim=-1) - q.norm(dim=-1)).abs().max()):.2e}) — the same "
        "angle must drive the pair (x_i, x_{i+d/2})")


def check_repeat_kv_expands_each_head_in_place(ns):
    """Head i of the expanded tensor must be KV head i // n_rep. Interleaving them
    the other way silently pairs every query head with the wrong KV head."""
    x = torch.randn(B, KV_HEADS, S, HEAD_DIM)
    n_rep = HEADS // KV_HEADS
    out = ns.repeat_kv(x, n_rep)
    assert tuple(out.shape) == (B, KV_HEADS * n_rep, S, HEAD_DIM), \
        f"expected {(B, KV_HEADS * n_rep, S, HEAD_DIM)}, got {tuple(out.shape)}"
    for i in range(KV_HEADS * n_rep):
        assert torch.allclose(out[:, i], x[:, i // n_rep], atol=1e-6), (
            f"expanded head {i} should be a copy of KV head {i // n_rep}; it matches "
            f"{[j for j in range(KV_HEADS) if torch.allclose(out[:, i], x[:, j])] or 'none'}"
            " instead")


def check_repeat_kv_identity_when_n_rep_is_one(ns):
    x = torch.randn(B, KV_HEADS, S, HEAD_DIM)
    out = ns.repeat_kv(x, 1)
    assert torch.allclose(out, x, atol=1e-6), \
        "repeat_kv(x, 1) must return x unchanged"


def check_rotary_embedder_lies_on_unit_circle(ns):
    torch.manual_seed(0)
    emb = ns.RotaryEmbedder(dim=HEAD_DIM, base=10000.0)
    cos, sin = emb(torch.randn(B, HEADS, S, HEAD_DIM))
    assert cos.shape[-1] == HEAD_DIM and sin.shape[-1] == HEAD_DIM, (
        f"cos/sin should end in the head dim {HEAD_DIM}, got {tuple(cos.shape)} "
        f"and {tuple(sin.shape)}")
    assert cos.shape[-2] == S and sin.shape[-2] == S, \
        f"expected one (cos, sin) row per position ({S}), got {tuple(cos.shape)}"
    unit = cos.pow(2) + sin.pow(2)
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-5), (
        "cos^2 + sin^2 must be 1 everywhere (max deviation "
        f"{float((unit - 1).abs().max()):.2e}) — these are angles on the unit circle")


def check_rotary_embedder_starts_at_angle_zero(ns):
    """Position 0 has angle 0, so it is the identity rotation. Off-by-one position
    indexing shows up here."""
    torch.manual_seed(0)
    emb = ns.RotaryEmbedder(dim=HEAD_DIM, base=10000.0)
    cos, sin = emb(torch.randn(B, HEADS, S, HEAD_DIM))
    c0 = cos.reshape(-1, S, HEAD_DIM)[0, 0]
    s0 = sin.reshape(-1, S, HEAD_DIM)[0, 0]
    assert torch.allclose(c0, torch.ones_like(c0), atol=1e-5), \
        f"cos at position 0 should be all 1 (angle 0), got {c0.tolist()[:4]}..."
    assert torch.allclose(s0, torch.zeros_like(s0), atol=1e-5), \
        f"sin at position 0 should be all 0 (angle 0), got {s0.tolist()[:4]}..."


def check_attention_output_shape(ns):
    attn = _attn(ns)
    x = torch.randn(B, S, HIDDEN)
    out = attn(x, _causal())
    out = out[0] if isinstance(out, tuple) else out
    assert tuple(out.shape) == (B, S, HIDDEN), \
        f"expected {(B, S, HIDDEN)}, got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "attention output contains NaN or inf"


def check_attention_is_position_aware(ns):
    """The whole point of RoPE. Without it, attention is permutation-equivariant:
    shuffling the sequence would just shuffle the output rows."""
    attn = _attn(ns, seed=1)
    torch.manual_seed(2)
    x = torch.randn(1, S, HIDDEN)
    free = torch.zeros(1, 1, S, S)          # no mask, so only position encoding can break symmetry
    perm = torch.tensor([2, 0, 5, 1, 4, 3])

    with torch.no_grad():
        base = attn(x, free)
        shuf = attn(x[:, perm], free)
    base = base[0] if isinstance(base, tuple) else base
    shuf = shuf[0] if isinstance(shuf, tuple) else shuf

    assert not torch.allclose(shuf, base[:, perm], atol=1e-5), (
        "permuting the input sequence just permuted the output — the block has no "
        "position information. Are the rotary embeddings applied to q and k?")


def check_attention_respects_the_causal_mask(ns):
    """With a -inf upper triangle, position t cannot see t+1.., so changing the
    last token must leave every earlier output row alone."""
    attn = _attn(ns, seed=3)
    torch.manual_seed(4)
    x = torch.randn(B, S, HIDDEN)
    mask = _causal()

    x2 = x.clone()
    x2[:, -1] += 5.0

    with torch.no_grad():
        base = attn(x, mask)
        moved = attn(x2, mask)
    base = base[0] if isinstance(base, tuple) else base
    moved = moved[0] if isinstance(moved, tuple) else moved

    drift = (base[:, :-1] - moved[:, :-1]).abs().max()
    assert drift < 1e-5, (
        f"changing the LAST token moved earlier outputs (max drift {float(drift):.2e}) "
        "— the additive mask must be added to the scores before the softmax")


def check_attention_uses_grouped_kv(ns):
    """GQA's reason to exist: with kv_heads < num_heads the K/V projections are
    narrower, so the block has fewer parameters than full multi-head attention."""
    attn = _attn(ns)
    n = sum(p.numel() for p in attn.parameters())
    mha = 4 * HIDDEN * HIDDEN
    assert n > 0, "RopeAttention has no parameters — where are the Q/K/V/O projections?"
    assert n < mha, (
        f"the block has {n} parameters, which is not fewer than the {mha} a full "
        f"multi-head attention block needs at hidden_size={HIDDEN}. With "
        f"kv_heads={KV_HEADS} < num_heads={HEADS}, W_key and W_value should each "
        f"produce only {KV_HEADS * HEAD_DIM} features, not {HIDDEN}")


def check_attention_gradients_flow(ns):
    attn = _attn(ns, seed=5)
    torch.manual_seed(6)
    x = torch.randn(B, S, HIDDEN, requires_grad=True)
    out = attn(x, _causal())
    (out[0] if isinstance(out, tuple) else out).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0, \
        "no gradient reached the input — is the graph broken by a detach/no_grad?"
    dead = [n for n, p in attn.named_parameters()
            if p.grad is None or float(p.grad.abs().sum()) == 0]
    assert not dead, f"no gradient reached: {', '.join(dead)} — unused projections?"


CHECKS = [check_rotate_half_negates_when_applied_twice,
          check_rotary_pos_emb_identity,
          check_rotary_pos_emb_preserves_norm,
          check_repeat_kv_expands_each_head_in_place,
          check_repeat_kv_identity_when_n_rep_is_one,
          check_rotary_embedder_lies_on_unit_circle,
          check_rotary_embedder_starts_at_angle_zero,
          check_attention_output_shape,
          check_attention_is_position_aware,
          check_attention_respects_the_causal_mask,
          check_attention_uses_grouped_kv,
          check_attention_gradients_flow]
