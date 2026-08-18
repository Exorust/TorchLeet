"""transformer — positional encoding, multi-head self-attention, FFN, block, model.

None of these are diffed against a reference; the checks lean on the structural
symmetries that define each piece and that a broken implementation breaks:

* positional encoding adds a table that does not depend on x, and for sinusoidal
  encodings PE(p)·PE(q) depends only on the offset p-q — true for either
  sin/cos interleaving convention, false for a learned or mis-scaled table;
* self-attention is permutation-equivariant and must not leak across the batch —
  the two failure modes of a wrong reshape/transpose in the head split;
* the feed-forward block is position-wise (per token) and genuinely nonlinear.
"""
import torch
import torch.nn as nn

ENTRIES = ["PositionalEncoding", "MultiHeadSelfAttention", "FeedForward",
           "TransformerEncoderLayer", "TransformerModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Shapes: everything below the model works on (batch, seq_len, embed_dim) and "
    "returns the same shape.",
    "In attention, split the last dim into (num_heads, head_dim) and move heads "
    "next to the batch: (B, H, T, head_dim). Scale the scores by "
    "1/sqrt(head_dim) before the softmax over the *last* axis.",
    "PositionalEncoding builds a (max_len, d_model) table once with "
    "sin(pos/10000^(2i/d)) and cos(...), registers it as a buffer, and forward "
    "just adds the first seq_len rows to x.",
]

B, T, D, H, FF = 3, 7, 16, 4, 32
VOCAB, LAYERS, OUT = 23, 2, 5


def _pos(ns, d_model=D):
    torch.manual_seed(0)
    return ns.PositionalEncoding(d_model)


def _attn(ns, embed=D, heads=H):
    torch.manual_seed(0)
    return ns.MultiHeadSelfAttention(embed, heads)


def _ffn(ns, embed=D, ff=FF):
    torch.manual_seed(0)
    return ns.FeedForward(embed, ff)


def _layer(ns, embed=D, heads=H, ff=FF):
    torch.manual_seed(0)
    return ns.TransformerEncoderLayer(embed, heads, ff)


def _model(ns, seed=0):
    torch.manual_seed(seed)
    return ns.TransformerModel(VOCAB, D, H, LAYERS, FF, OUT)


# --------------------------------------------------------------- positional


def check_positional_encoding_preserves_shape(ns):
    pe = _pos(ns)
    for t in (1, T, 40):
        out = pe(torch.zeros(B, t, D))
        assert tuple(out.shape) == (B, t, D), (
            f"seq_len {t}: expected {(B, t, D)} back, got {tuple(out.shape)} — "
            "the encoding is added to x, it does not change the shape")


def check_positional_encoding_is_added_not_computed(ns):
    """pe(x) - x must be the same table whatever x is, and shared across the batch."""
    pe = _pos(ns)
    pe.eval()
    x1, x2 = torch.randn(B, T, D), torch.randn(B, T, D) * 7
    with torch.no_grad():
        d1, d2 = pe(x1) - x1, pe(x2) - x2
    assert torch.allclose(d1, d2, atol=1e-5), (
        f"the offset added to x changed with x (max diff "
        f"{(d1 - d2).abs().max():.2e}) — a positional encoding depends only on "
        "position, so forward should be `x + self.pe[:, :seq_len]`")
    assert torch.allclose(d1[0], d1[1], atol=1e-6), \
        "different rows of the batch got different positional encodings"
    assert not torch.allclose(d1[0, 0], d1[0, 1], atol=1e-6), \
        "position 0 and position 1 got the same encoding — there is no order signal"


def check_positional_encoding_is_sinusoidal(ns):
    """Sinusoidal encodings are bounded and translation-invariant.

    PE(p)·PE(q) = sum_i cos((p-q) * w_i) depends only on the offset, whichever
    way sin and cos are interleaved. A random or learned table has no such
    structure.
    """
    pe = _pos(ns, d_model=32)
    pe.eval()
    n = 24
    with torch.no_grad():
        table = (pe(torch.zeros(1, n, 32)) - torch.zeros(1, n, 32))[0]
    assert float(table.abs().max()) <= 1.0 + 1e-4, (
        f"encoding values reach {float(table.abs().max()):.3f}; sin and cos stay "
        "inside [-1, 1]")
    for offset in (1, 5):
        dots = torch.stack([table[p] @ table[p + offset] for p in range(0, 12)])
        spread = float(dots.max() - dots.min())
        assert spread < 1e-2 * max(1.0, float(dots.abs().mean())) + 1e-3, (
            f"PE(p)·PE(p+{offset}) ranges over {spread:.3e} as p moves; for "
            "sinusoidal encodings it is constant — check the 1/10000^(2i/d) "
            "frequencies and that sin and cos share each frequency")


# --------------------------------------------------------------- attention


def check_attention_preserves_shape(ns):
    for heads in (1, 2, 4):
        a = _attn(ns, heads=heads)
        out = a(torch.randn(B, T, D))
        assert tuple(out.shape) == (B, T, D), (
            f"num_heads={heads}: expected {(B, T, D)} back, got "
            f"{tuple(out.shape)} — concatenate the heads and project back to "
            "embed_dim")


def check_attention_does_not_mix_the_batch(ns):
    """Sequence i's output must not depend on sequence j — the classic reshape bug."""
    a = _attn(ns)
    a.eval()
    x = torch.randn(B, T, D)
    with torch.no_grad():
        together = a(x)
        alone = torch.cat([a(x[i:i + 1]) for i in range(B)], dim=0)
    assert torch.allclose(together, alone, atol=1e-5), (
        f"running a batch of {B} differs from running each sequence on its own "
        f"(max diff {(together - alone).abs().max():.2e}) — the head split is "
        "folding the batch into another axis")


def check_attention_is_permutation_equivariant(ns):
    """Self-attention has no notion of order: shuffle the tokens, the outputs
    shuffle with them. Fails if the scores or the softmax use the wrong axis."""
    a = _attn(ns)
    a.eval()
    x = torch.randn(1, T, D)
    perm = torch.randperm(T)
    with torch.no_grad():
        shuffled_out = a(x[:, perm])
        out_shuffled = a(x)[:, perm]
    assert torch.allclose(shuffled_out, out_shuffled, atol=1e-5), (
        f"attention(x[perm]) != attention(x)[perm] (max diff "
        f"{(shuffled_out - out_shuffled).abs().max():.2e}) — softmax over the "
        "wrong dimension, or q/k/v got mixed up in the reshape")


def check_attention_mixes_across_tokens(ns):
    """It must actually attend: changing one token moves the others' outputs."""
    a = _attn(ns)
    a.eval()
    x = torch.randn(1, T, D)
    y = x.clone()
    y[0, 0] += 10.0
    with torch.no_grad():
        out_x, out_y = a(x), a(y)
    moved = (out_x[0, 1:] - out_y[0, 1:]).abs().max()
    assert float(moved) > 1e-6, (
        "changing token 0 left every other token's output untouched — the "
        "attention weights are not carrying information between positions")


def check_attention_softmax_normalizes_over_keys(ns):
    """Pins which axis the softmax runs over.

    Permutation equivariance cannot see this: softmax over the query axis is
    equivariant too. But if every *column* sums to 1 then summing the outputs
    over tokens collapses to sum_j v_j, which is affine in x — so
    g(2x) - g(0) would be exactly 2*(g(x) - g(0)). With the correct per-query
    normalisation the weights themselves move when x is rescaled, and that
    identity breaks by a wide margin.
    """
    a = _attn(ns)
    a.eval()
    torch.manual_seed(3)
    x = torch.randn(1, T, D)
    with torch.no_grad():
        g0 = a(torch.zeros(1, T, D)).sum(dim=1)
        g1 = a(x).sum(dim=1)
        g2 = a(2 * x).sum(dim=1)
    residual = float(((g2 - g0) - 2 * (g1 - g0)).abs().max())
    scale = max(float((g1 - g0).abs().max()), 1e-3)
    assert residual > 1e-3 * scale, (
        f"summing the outputs over tokens turned out to be affine in x "
        f"(residual {residual:.2e} against a scale of {scale:.2e}) — the softmax "
        "is normalising each key over the queries instead of each query over the "
        "keys; it belongs on the last axis of the (B, heads, T_query, T_key) "
        "score tensor")


def check_attention_gradients_flow(ns):
    a = _attn(ns)
    a(torch.randn(2, T, D)).sum().backward()
    dead = [n for n, p in a.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — the q/k/v projection and the output "
        "projection should both be on the path")


# --------------------------------------------------------------- feed-forward


def check_feedforward_is_position_wise(ns):
    """The FFN sees one token at a time, so tokens are independent."""
    f = _ffn(ns)
    f.eval()
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out = f(x)
        assert tuple(out.shape) == (B, T, D), (
            f"expected {(B, T, D)} back (embed_dim -> ff_dim -> embed_dim), got "
            f"{tuple(out.shape)}")
        one_token = f(x[:, 2:3])
    assert torch.allclose(out[:, 2:3], one_token, atol=1e-5), (
        f"a token's output changed when its neighbours were removed (max diff "
        f"{(out[:, 2:3] - one_token).abs().max():.2e}) — the feed-forward block "
        "is applied per position, it must not look across the sequence")


def check_feedforward_is_nonlinear(ns):
    """Two Linears with no activation between them collapse into one Linear."""
    f = _ffn(ns)
    f.eval()
    a = torch.randn(1, 1, D)
    b = torch.randn(1, 1, D)
    with torch.no_grad():
        gap = f(a + b) - f(a) - f(b) + f(torch.zeros(1, 1, D))
    assert float(gap.abs().max()) > 1e-4, (
        "f(a+b) - f(a) - f(b) + f(0) is zero, so the block is affine — the ReLU "
        "between the two Linear layers is missing")


def check_feedforward_gradients_flow(ns):
    f = _ffn(ns)
    f(torch.randn(2, T, D)).sum().backward()
    dead = [n for n, p in f.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''}")


# --------------------------------------------------------------- encoder layer


def check_encoder_layer_preserves_shape(ns):
    layer = _layer(ns)
    out = layer(torch.randn(B, T, D))
    assert tuple(out.shape) == (B, T, D), (
        f"an encoder layer is shape-preserving: expected {(B, T, D)}, got "
        f"{tuple(out.shape)}")


def check_encoder_layer_is_permutation_equivariant(ns):
    layer = _layer(ns)
    layer.eval()
    x = torch.randn(1, T, D)
    perm = torch.randperm(T)
    with torch.no_grad():
        shuffled_out, out_shuffled = layer(x[:, perm]), layer(x)[:, perm]
    assert torch.allclose(shuffled_out, out_shuffled, atol=1e-5), (
        f"layer(x[perm]) != layer(x)[perm] (max diff "
        f"{(shuffled_out - out_shuffled).abs().max():.2e}) — every sublayer works "
        "per token or symmetrically over tokens, so the whole layer must too")


def check_encoder_layer_gradients_flow(ns):
    layer = _layer(ns)
    layer(torch.randn(2, T, D)).sum().backward()
    dead = [n for n, p in layer.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — attention, feed-forward and both "
        "LayerNorms should all be on the forward path")


def check_encoder_layer_normalizes(ns):
    """LayerNorm plus residuals keep the block stable on a huge input."""
    layer = _layer(ns)
    layer.eval()
    with torch.no_grad():
        out = layer(torch.randn(2, T, D) * 1e4)
    assert not torch.isnan(out).any() and not torch.isinf(out).any(), \
        "NaN or Inf on a large input — the softmax or the norm is unstable"


# --------------------------------------------------------------- whole model


def check_model_output_shape(ns):
    m = _model(ns)
    m.eval()
    out = m(torch.randint(0, VOCAB, (B, T)))
    assert out.ndim == 2, (
        f"expected a 2D (batch, output_dim) prediction, got a {out.ndim}D tensor "
        f"{tuple(out.shape)} — pool the sequence (CLS token or mean) before the "
        "output projection")
    assert tuple(out.shape) == (B, OUT), \
        f"expected {(B, OUT)} for a ({B}, {T}) batch of token ids, got {tuple(out.shape)}"


def check_model_accepts_any_batch_and_length(ns):
    m = _model(ns)
    m.eval()
    for b, t in ((1, 1), (4, 9), (2, 30)):
        out = m(torch.randint(0, VOCAB, (b, t)))
        assert tuple(out.shape) == (b, OUT), (
            f"a ({b}, {t}) batch of ids gave {tuple(out.shape)}, expected "
            f"{(b, OUT)} — the model must handle variable-length padded input")


def check_model_gradients_reach_all_parameters(ns):
    m = _model(ns)
    out = m(torch.randint(0, VOCAB, (2, T)))
    nn.CrossEntropyLoss()(out, torch.tensor([0, 1])).backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — every encoder layer, the embedding "
        "and the output projection should be on the forward path")


def check_model_can_learn(ns):
    """Overfit a fixed batch of ids; loss must fall."""
    torch.manual_seed(0)
    m = _model(ns)
    x = torch.randint(0, VOCAB, (8, T))
    y = torch.randint(0, OUT, (8,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    first = lossf(m(x), y).item()
    loss = None
    for _ in range(40):
        opt.zero_grad()
        loss = lossf(m(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first, (
        f"loss did not decrease over 40 steps on a fixed batch "
        f"({first:.4f} -> {loss.item():.4f}) — the model is not learning")


CHECKS = [
    check_positional_encoding_preserves_shape,
    check_positional_encoding_is_added_not_computed,
    check_positional_encoding_is_sinusoidal,
    check_attention_preserves_shape,
    check_attention_does_not_mix_the_batch,
    check_attention_is_permutation_equivariant,
    check_attention_mixes_across_tokens,
    check_attention_softmax_normalizes_over_keys,
    check_attention_gradients_flow,
    check_feedforward_is_position_wise,
    check_feedforward_is_nonlinear,
    check_feedforward_gradients_flow,
    check_encoder_layer_preserves_shape,
    check_encoder_layer_is_permutation_equivariant,
    check_encoder_layer_gradients_flow,
    check_encoder_layer_normalizes,
    check_model_output_shape,
    check_model_accepts_any_batch_and_length,
    check_model_gradients_reach_all_parameters,
    check_model_can_learn,
]
