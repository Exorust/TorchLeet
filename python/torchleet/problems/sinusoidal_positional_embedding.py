"""sinusoidal-positional-embedding."""
import math
import torch

ENTRIES = ["SinusoidalPositionalEmbedding"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Even dimensions 2i get sin and odd dimensions 2i+1 get cos. The pair shares one frequency.",
    "i is 0, 1, ..., d_model/2 - 1, not the raw dimension. The frequency is 1 / 10000**(2i/d_model).",
    "It is a fixed function of position — register it as a buffer, not a parameter.",
]

MAX_LEN, D_MODEL = 32, 64


def _pe(ns):
    torch.manual_seed(0)
    return ns.SinusoidalPositionalEmbedding(MAX_LEN, D_MODEL)


def _table(m):
    """Get the (max_len, d_model) table however the module exposes it."""
    for attr in ("pe", "encoding", "embeddings", "weight"):
        t = getattr(m, attr, None)
        if torch.is_tensor(t):
            return t.squeeze()
    out = m(torch.zeros(1, MAX_LEN, D_MODEL))
    return (out[0] if out.dim() == 3 else out).squeeze()


def check_table_shape(ns):
    t = _table(_pe(ns))
    assert t.shape[-1] == D_MODEL, \
        f"embedding dimension should be {D_MODEL}, got {t.shape[-1]}"
    assert t.shape[-2] >= MAX_LEN or t.shape[0] >= MAX_LEN, \
        f"expected at least {MAX_LEN} positions, got {tuple(t.shape)}"


def check_values_bounded(ns):
    t = _table(_pe(ns))
    assert not torch.isnan(t).any(), "NaN in the embedding table"
    assert float(t.abs().max()) <= 1.0 + 1e-5, \
        f"sin/cos must stay in [-1, 1], got max |value| {float(t.abs().max()):.3f}"


def check_deterministic(ns):
    a, b = _table(_pe(ns)), _table(_pe(ns))
    assert torch.allclose(a, b), \
        "the table is not a fixed function of position — is it randomly initialized?"


def check_positions_distinct(ns):
    t = _table(_pe(ns))[:MAX_LEN]
    d = torch.cdist(t.unsqueeze(0), t.unsqueeze(0)).squeeze(0)
    d = d + torch.eye(d.shape[0]) * 10
    assert float(d.min()) > 1e-6, "two positions have identical embeddings"


def check_sinusoidal_values(ns):
    """Even dims are sin(pos * ω_i), odd dims are cos(pos * ω_i), with the
    paper's ω_i = 1/10000**(2i/d_model) shared by dimension pair (2i, 2i+1)."""
    t = _table(_pe(ns))
    assert t.dim() == 2 and t.shape[0] >= MAX_LEN and t.shape[-1] == D_MODEL, \
        f"expected a table of shape (position, {D_MODEL}), got {tuple(t.shape)}"
    t = t[:MAX_LEN, :D_MODEL].float()
    omega = torch.exp(
        torch.arange(0, D_MODEL, 2).float() * (-math.log(10000.0) / D_MODEL))
    pos = torch.arange(MAX_LEN, dtype=torch.float).unsqueeze(1)
    assert torch.allclose(t[:, 0::2], torch.sin(pos * omega), atol=1e-5), \
        "even dimensions 2i should be sin(pos / 10000**(2i/d_model)); " \
        "i is the pair index 0, 1, ..., d_model/2 - 1"
    assert torch.allclose(t[:, 1::2], torch.cos(pos * omega), atol=1e-5), \
        "odd dimensions 2i+1 should be cos of that same angle, not a new frequency"


def check_forward_slices_seq_len(ns):
    m = _pe(ns)
    seq, batch = 7, 4
    out = m(torch.zeros(batch, seq, D_MODEL))
    assert tuple(out.shape) == (1, seq, D_MODEL), \
        "forward(x) should return shape (1, seq_len, d_model) so it broadcasts " \
        f"over the batch; got {tuple(out.shape)}"
    table = _table(m)[:seq, :D_MODEL].float()
    assert torch.allclose(out[0].float(), table, atol=1e-5), \
        "forward should return the first seq_len rows of the positional table " \
        "and ignore the values in x"


CHECKS = [check_table_shape, check_values_bounded, check_deterministic,
          check_positions_distinct, check_sinusoidal_values,
          check_forward_slices_seq_len]
