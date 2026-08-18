"""2d-positional-embeddings — sinusoidal position encodings for image grids."""
import torch

ENTRIES = ["create_2d_sinusoidal_embeddings"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Encode the row and the column separately, then concatenate the two halves.",
    "Each half is a standard 1D sinusoidal embedding of size d_model // 2.",
    "Flatten the grid in row-major order so index r*width + c is position (r, c).",
]

H, W, D = 4, 6, 32


def _emb(ns):
    return ns.create_2d_sinusoidal_embeddings(H, W, D)


def check_shape(ns):
    e = _emb(ns)
    assert tuple(e.shape) == (H * W, D), \
        f"expected {(H * W, D)} (flattened grid), got {tuple(e.shape)}"


def check_deterministic(ns):
    assert torch.allclose(_emb(ns), _emb(ns)), \
        "embeddings must be a fixed function of position, not random"


def check_values_are_bounded(ns):
    e = _emb(ns)
    assert not torch.isnan(e).any(), "NaN in embeddings"
    assert float(e.abs().max()) <= 1.0 + 1e-6, \
        f"sin/cos outputs must stay within [-1, 1], got max |value| {float(e.abs().max()):.3f}"


def check_positions_are_distinct(ns):
    """Different grid cells must not collide, or attention cannot tell them apart."""
    e = _emb(ns)
    dup = torch.cdist(e, e) + torch.eye(H * W) * 10
    assert float(dup.min()) > 1e-6, \
        "two different grid positions produced identical embeddings"


def check_encodes_both_axes(ns):
    """Moving along a row and along a column must both change the embedding."""
    e = _emb(ns).reshape(H, W, D)
    assert not torch.allclose(e[0, 0], e[0, 1], atol=1e-6), \
        "changing the column did not change the embedding — width is not encoded"
    assert not torch.allclose(e[0, 0], e[1, 0], atol=1e-6), \
        "changing the row did not change the embedding — height is not encoded"


CHECKS = [check_shape, check_deterministic, check_values_are_bounded,
          check_positions_are_distinct, check_encodes_both_axes]
