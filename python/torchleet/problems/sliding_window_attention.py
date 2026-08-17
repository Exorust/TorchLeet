"""sliding-window-attention.

Self-consistency anchor: with a window wide enough to cover the whole sequence,
sliding-window attention must reduce to full attention. That check uses the
solver's own full_attention, so it holds whatever scaling or masking style they chose.
"""
import torch

ENTRIES = ["create_sliding_window_mask", "full_attention", "sliding_window_attention"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Position i may attend to j when |i - j| <= window_size.",
    "Build the mask from pairwise distances: (arange - arange.T).abs() <= window_size.",
    "Masked positions get -inf BEFORE the softmax, not zero after it.",
]

B, S, D = 2, 12, 16


def _qkv(seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, S, D) for _ in range(3))


def check_mask_shape_and_dtype(ns):
    m = ns.create_sliding_window_mask(S, 2)
    assert tuple(m.shape) == (S, S), f"mask should be {(S, S)}, got {tuple(m.shape)}"
    assert m.dtype == torch.bool, f"mask should be boolean, got {m.dtype}"


def check_mask_marks_the_right_positions(ns):
    w = 2
    m = ns.create_sliding_window_mask(S, w)
    idx = torch.arange(S)
    expected = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= w
    assert torch.equal(m, expected), (
        "mask does not match |i - j| <= window_size "
        f"({int((m != expected).sum())} positions differ)")


def check_diagonal_always_visible(ns):
    m = ns.create_sliding_window_mask(S, 0)
    assert bool(m.diagonal().all()), "every position must attend to itself"


def check_output_shape(ns):
    q, k, v = _qkv()
    out = ns.sliding_window_attention(q, k, v, window_size=3)
    assert tuple(out.shape) == (B, S, D), \
        f"expected {(B, S, D)}, got {tuple(out.shape)}"


def check_wide_window_equals_full_attention(ns):
    """Window >= seq_len masks nothing, so it must equal full attention."""
    q, k, v = _qkv(1)
    wide = ns.sliding_window_attention(q, k, v, window_size=S)
    full = ns.full_attention(q, k, v)
    diff = (wide - full).abs().max().item()
    assert torch.allclose(wide, full, atol=1e-5), (
        f"with window_size >= seq_len nothing is masked, so this must equal "
        f"full_attention (max diff {diff:.2e})")


def check_narrow_window_differs_from_full(ns):
    """Guards against ignoring the mask entirely."""
    q, k, v = _qkv(2)
    narrow = ns.sliding_window_attention(q, k, v, window_size=1)
    full = ns.full_attention(q, k, v)
    assert not torch.allclose(narrow, full, atol=1e-4), \
        "window_size=1 gives the same result as full attention — the mask is not applied"


CHECKS = [check_mask_shape_and_dtype, check_mask_marks_the_right_positions,
          check_diagonal_always_visible, check_output_shape,
          check_wide_window_equals_full_attention, check_narrow_window_differs_from_full]
