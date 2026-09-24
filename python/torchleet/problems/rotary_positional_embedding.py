"""rotary-positional-embedding (RoPE).

RoPE is a rotation, which gives two exact properties to test: applying
rotate_half twice negates the vector, and rotating with a unit (cos, sin) pair
preserves the norm. Neither of those, nor the zero-angle check, can tell a real
rotation from returning q and k untouched, so check_quarter_turn pins the angle:
at cos=0, sin=1 the result must be exactly rotate_half of the input.
"""
import torch

ENTRIES = ["rotate_half", "apply_rotary_pos_emb"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "rotate_half splits the last dim in two and returns (-x2, x1) concatenated.",
    "Rotated q = q*cos + rotate_half(q)*sin — the 2D rotation written elementwise.",
    "cos and sin must broadcast against (batch, heads, seq, dim).",
]

B, H, S, D = 2, 4, 8, 16


def check_rotate_half_shape(ns):
    x = torch.randn(B, H, S, D)
    assert tuple(ns.rotate_half(x).shape) == (B, H, S, D), \
        f"rotate_half must preserve shape, got {tuple(ns.rotate_half(x).shape)}"


def check_rotate_half_twice_negates(ns):
    """(-x2, x1) applied twice gives (-x1, -x2) = -x."""
    x = torch.randn(B, H, S, D)
    twice = ns.rotate_half(ns.rotate_half(x))
    assert torch.allclose(twice, -x, atol=1e-6), \
        "applying rotate_half twice should negate the input"


def check_rotate_half_permutes_values(ns):
    x = torch.randn(2, 4)
    out = ns.rotate_half(x)
    assert torch.allclose(out.abs().sort(-1).values, x.abs().sort(-1).values, atol=1e-6), \
        "rotate_half should move values around and flip signs, not change magnitudes"


def check_identity_rotation(ns):
    """cos=1, sin=0 is a zero-angle rotation: q and k come back unchanged."""
    q, k = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    cos, sin = torch.ones(S, D), torch.zeros(S, D)
    rq, rk = ns.apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(rq, q, atol=1e-6), "cos=1, sin=0 must leave q unchanged"
    assert torch.allclose(rk, k, atol=1e-6), "cos=1, sin=0 must leave k unchanged"


def check_quarter_turn(ns):
    """cos=0, sin=1 is a 90-degree turn: q*0 + rotate_half(q)*1 = rotate_half(q).

    Uses the solver's own rotate_half, so any consistent pairing convention passes,
    but returning q or k unrotated (or ignoring sin) does not.
    """
    torch.manual_seed(1)
    q, k = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    cos, sin = torch.zeros(S, D), torch.ones(S, D)
    rq, rk = ns.apply_rotary_pos_emb(q, k, cos, sin)
    assert not torch.allclose(rq, q, atol=1e-4), (
        "with cos=0, sin=1 the query came back unchanged — apply_rotary_pos_emb "
        "must actually use cos and sin (q*cos + rotate_half(q)*sin)")
    assert torch.allclose(rq, ns.rotate_half(q), atol=1e-6), (
        "with cos=0, sin=1 the rotated query should equal rotate_half(q) — "
        "check the q*cos + rotate_half(q)*sin formula")
    assert torch.allclose(rk, ns.rotate_half(k), atol=1e-6), (
        "with cos=0, sin=1 the rotated key should equal rotate_half(k) — the key "
        "must be rotated exactly like the query")


def check_rotation_preserves_norm(ns):
    """A rotation cannot change a vector's length."""
    torch.manual_seed(0)
    q, k = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    ang = torch.rand(S, D // 2) * 6.28
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    rq, _ = ns.apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-4), \
        "rotation changed the vector norms — check the cos/sin pairing"


CHECKS = [check_rotate_half_shape, check_rotate_half_twice_negates,
          check_rotate_half_permutes_values, check_identity_rotation,
          check_quarter_turn, check_rotation_preserves_norm]
