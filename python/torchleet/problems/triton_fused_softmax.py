"""triton-fused-softmax — online (single-pass) softmax, then a Triton kernel.

Device policy: the PyTorch part is real work and is checkable on CPU, so it is
graded everywhere. The Triton kernel needs a GPU and is reported SKIPPED without
one — never passed. A CPU run therefore reports partial verification, which is
the honest answer; it does not claim the kernel is correct.
"""
import torch

from torchleet.runner import Skip

ENTRIES = ["online_softmax_pytorch"]
DEVICE = "cpu"          # the CPU-checkable portion; the kernel check gates itself
EXTRAS = []             # triton is only needed by the optional kernel check

HINTS = [
    "Online softmax keeps a running max and running sum in one pass.",
    "When a new max arrives, rescale the accumulated sum by exp(old_max - new_max).",
    "Subtracting the running max before exp() is what keeps it stable on large inputs.",
]


def check_matches_torch_softmax(ns):
    """torch.softmax is the oracle — it already ships with the user's PyTorch."""
    torch.manual_seed(0)
    for shape in [(4, 8), (1, 128), (16, 33)]:
        x = torch.randn(*shape)
        got = ns.online_softmax_pytorch(x)
        exp = torch.softmax(x, dim=-1)
        assert got.shape == exp.shape, \
            f"shape {tuple(got.shape)} != expected {tuple(exp.shape)} for input {shape}"
        assert torch.allclose(got, exp, atol=1e-6), (
            f"disagrees with torch.softmax on input {shape} "
            f"(max diff {(got - exp).abs().max().item():.2e})")


def check_rows_sum_to_one(ns):
    x = torch.randn(8, 64)
    out = ns.online_softmax_pytorch(x)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), \
        f"softmax rows must sum to 1, got range [{sums.min():.4f}, {sums.max():.4f}]"


def check_numerically_stable(ns):
    """The point of online softmax: large inputs must not overflow."""
    x = torch.tensor([[1e4, 1e4 + 1.0, 1e4 + 2.0], [-1e4, -1e4 - 1.0, -1e4 - 2.0]])
    out = ns.online_softmax_pytorch(x)
    assert not torch.isnan(out).any(), "NaN on large inputs — subtract the running max"
    assert not torch.isinf(out).any(), "Inf on large inputs — subtract the running max"
    assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6), \
        "wrong values on large inputs (numerical stability)"


def check_triton_kernel(ns):
    """GPU-only. Skipped, never passed, when it cannot actually be verified."""
    if not torch.cuda.is_available():
        raise Skip("requires CUDA — Triton kernel not verified on this machine")
    try:
        import triton  # noqa: F401
    except ImportError:
        raise Skip("needs `triton` — pip install triton")
    fn = getattr(ns, "fused_softmax_triton", None)
    if fn is None:
        raise Skip("pass fused_softmax_triton=... to check the kernel")
    x = torch.randn(8, 512, device="cuda")
    got = fn(x)
    assert torch.allclose(got, torch.softmax(x, dim=-1), atol=1e-5), \
        f"Triton kernel disagrees with torch.softmax (max diff {(got - torch.softmax(x, dim=-1)).abs().max().item():.2e})"


CHECKS = [
    check_matches_torch_softmax,
    check_rows_sum_to_one,
    check_numerically_stable,
    check_triton_kernel,
]
