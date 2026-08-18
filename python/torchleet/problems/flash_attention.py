"""flash-attention — tiled attention with an online softmax.

The graded entry is the PyTorch implementation, which runs anywhere. The Triton
kernel is checked too, but only when a GPU with Triton is actually present — it
is reported SKIPPED otherwise, never passed.

The load-bearing check is tile-size invariance: a correct online softmax gives
the same answer whatever block sizes it walks the keys in. An implementation
that forgets to rescale the accumulator by exp(prev_max - new_max) agrees with
the reference at one tile size and diverges at another, which a single-tile-size
comparison would miss entirely.
"""
import math

import torch
import torch.nn.functional as F

from torchleet.runner import Skip

ENTRIES = ["flash_attention_pytorch"]
DEVICE = "cpu"   # the PyTorch path is checkable anywhere; the kernel check self-gates
EXTRAS = []      # triton is only needed by the optional kernel check

HINTS = [
    "Keep a running max, a running denominator l, and a running output accumulator; "
    "on each K/V tile rescale all three by alpha = exp(prev_max - new_max).",
    "Divide the accumulator by l only once, after the loop — not inside it.",
    "L is the logsumexp of the scaled scores: running_max + log(l) at the end.",
]

B, N_Q, N_K, D = 2, 40, 48, 16


def _inputs(seed=0, scale=1.0):
    torch.manual_seed(seed)
    return (torch.randn(B, N_Q, D) * scale,
            torch.randn(B, N_K, D) * scale,
            torch.randn(B, N_K, D))


def _split(result):
    """Accept (O, L) or just O."""
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def check_output_matches_torch_attention(ns):
    q, k, v = _inputs()
    o, _ = _split(ns.flash_attention_pytorch(q, k, v))
    exp = F.scaled_dot_product_attention(q, k, v)
    assert tuple(o.shape) == tuple(exp.shape), \
        f"expected {tuple(exp.shape)}, got {tuple(o.shape)}"
    assert torch.allclose(o, exp, atol=1e-5), \
        f"disagrees with scaled_dot_product_attention (max diff {(o - exp).abs().max():.2e})"


def check_logsumexp_is_correct(ns):
    q, k, v = _inputs(1)
    _, lse = _split(ns.flash_attention_pytorch(q, k, v))
    if lse is None:
        raise Skip("this implementation returns only O, not (O, L)")
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    exp = torch.logsumexp(scores, dim=-1)
    assert torch.allclose(lse, exp, atol=1e-4), \
        f"L is not the logsumexp of the scaled scores (max diff {(lse - exp).abs().max():.2e})"


def check_result_is_independent_of_tile_size(ns):
    """The property that catches a missing online-softmax rescale."""
    q, k, v = _inputs(2)
    try:
        a, _ = _split(ns.flash_attention_pytorch(q, k, v, block_size_q=16, block_size_k=16))
        b, _ = _split(ns.flash_attention_pytorch(q, k, v, block_size_q=8, block_size_k=48))
    except TypeError:
        raise Skip("implementation does not expose block sizes")
    assert torch.allclose(a, b, atol=1e-5), (
        "different tile sizes gave different answers (max diff "
        f"{(a - b).abs().max():.2e}) — the running max/sum are not being rescaled "
        "by exp(prev_max - new_max) when a new block raises the maximum")


def check_numerically_stable(ns):
    """Large scores must not overflow — the reason for the running max."""
    q, k, v = _inputs(3, scale=30.0)
    o, _ = _split(ns.flash_attention_pytorch(q, k, v))
    assert not torch.isnan(o).any(), "NaN on large inputs — subtract the running max"
    assert not torch.isinf(o).any(), "Inf on large inputs — subtract the running max"
    exp = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(o, exp, atol=1e-4), "wrong values on large inputs"


def check_output_is_an_average_of_values(ns):
    """Attention output is a convex combination of V, so it cannot escape V's range."""
    q, k, v = _inputs(4)
    o, _ = _split(ns.flash_attention_pytorch(q, k, v))
    lo, hi = v.min().item(), v.max().item()
    assert o.min().item() >= lo - 1e-4 and o.max().item() <= hi + 1e-4, \
        f"output range [{o.min():.3f}, {o.max():.3f}] escapes V's range [{lo:.3f}, {hi:.3f}]"


def check_triton_kernel(ns):
    """GPU-only. Skipped, never passed, when it cannot actually be verified."""
    if not torch.cuda.is_available():
        raise Skip("requires CUDA — the Triton kernel is not verified on this machine")
    try:
        import triton
    except ImportError:
        raise Skip("needs `triton` — pip install triton")
    kernel = getattr(ns, "flash_fwd_kernel", None)
    if kernel is None:
        raise Skip("pass flash_fwd_kernel=... to check the kernel")

    torch.manual_seed(0)
    d = 64
    q = torch.randn(1, 64, d, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 128, d, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 128, d, dtype=torch.float16, device="cuda")
    o = torch.empty(1, 64, d, dtype=torch.float16, device="cuda")
    lse = torch.empty(1, 64, dtype=torch.float32, device="cuda")
    kernel[(triton.cdiv(64, 16), 1)](
        q, k, v, o, lse,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(), *lse.stride(),
        64, 128, scale=1.0 / math.sqrt(d), D=d,
        BLOCK_SIZE_Q=16, BLOCK_SIZE_K=16,
    )
    exp = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    assert torch.allclose(o.float(), exp, atol=1e-1, rtol=1e-2), \
        f"Triton kernel disagrees with torch attention (max diff {(o.float() - exp).abs().max():.2e})"


CHECKS = [check_output_matches_torch_attention, check_logsumexp_is_correct,
          check_result_is_independent_of_tile_size, check_numerically_stable,
          check_output_is_an_average_of_values, check_triton_kernel]
