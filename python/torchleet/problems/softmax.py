"""softmax — numerically stable softmax from scratch."""
import torch

ENTRIES = ["softmax"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "exp() of a large number overflows. What can you subtract without changing the result?",
    "softmax(x) == softmax(x - c) for any constant c. Pick c = max(x) along `dim`.",
    "Keep dims when you take the max and the sum so broadcasting lines up.",
]


def check_matches_torch_softmax(ns):
    torch.manual_seed(0)
    for shape in [(5,), (4, 8), (2, 3, 7)]:
        x = torch.randn(*shape)
        got, exp = ns.softmax(x), torch.softmax(x, dim=-1)
        assert got.shape == exp.shape, \
            f"shape {tuple(got.shape)} != {tuple(exp.shape)} for input {shape}"
        assert torch.allclose(got, exp, atol=1e-6), \
            f"disagrees with torch.softmax on {shape} (max diff {(got - exp).abs().max():.2e})"


def check_respects_dim_argument(ns):
    x = torch.randn(4, 6)
    got = ns.softmax(x, dim=0)
    assert torch.allclose(got, torch.softmax(x, dim=0), atol=1e-6), \
        "softmax(x, dim=0) should normalize down columns, not across rows"


def check_sums_to_one(ns):
    s = ns.softmax(torch.randn(6, 10)).sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6), \
        f"rows must sum to 1, got range [{s.min():.4f}, {s.max():.4f}]"


def check_numerically_stable(ns):
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    out = ns.softmax(x)
    assert not torch.isnan(out).any(), "NaN on large inputs — subtract the max first"
    assert not torch.isinf(out).any(), "Inf on large inputs — subtract the max first"
    assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6), \
        "wrong values on large inputs"


CHECKS = [check_matches_torch_softmax, check_respects_dim_argument,
          check_sums_to_one, check_numerically_stable]
