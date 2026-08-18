"""sinusoidal-positional-embedding."""
import torch

ENTRIES = ["SinusoidalPositionalEmbedding"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Even dimensions get sin, odd dimensions get cos.",
    "The frequency for dimension i is 1 / 10000**(2i/d_model).",
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


CHECKS = [check_table_shape, check_values_bounded, check_deterministic,
          check_positions_distinct]
