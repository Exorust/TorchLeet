"""knn — k-nearest neighbours in PyTorch."""
import torch

ENTRIES = ["knn_predict"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "torch.cdist gives every test-to-train distance in one call.",
    "topk(..., largest=False) picks the nearest neighbours.",
    "Predict the majority label among those neighbours (torch.mode helps).",
]


def _data(seed=0):
    torch.manual_seed(seed)
    Xtr = torch.cat([torch.randn(30, 2) - 3, torch.randn(30, 2) + 3])
    ytr = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    return Xtr, ytr


def check_output_shape_and_dtype(ns):
    Xtr, ytr = _data()
    Xte = torch.randn(7, 2)
    out = ns.knn_predict(Xtr, ytr, Xte, k=3)
    assert tuple(out.shape) == (7,), f"expected (7,), got {tuple(out.shape)}"
    assert set(out.tolist()) <= set(ytr.tolist()), "predicted a label not present in y_train"


def check_k_of_one_returns_nearest_label(ns):
    Xtr, ytr = _data()
    Xte = Xtr[:10] + 1e-4          # essentially on top of known points
    got = ns.knn_predict(Xtr, ytr, Xte, k=1)
    assert torch.equal(got, ytr[:10]), \
        "k=1 must return the label of the single nearest training point"


def check_separable_data_is_classified(ns):
    Xtr, ytr = _data()
    Xte = torch.cat([torch.randn(20, 2) - 3, torch.randn(20, 2) + 3])
    yte = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    acc = (ns.knn_predict(Xtr, ytr, Xte, k=5) == yte).float().mean()
    assert float(acc) > 0.9, \
        f"only {float(acc):.0%} accuracy on well-separated clusters"


CHECKS = [check_output_shape_and_dtype, check_k_of_one_returns_nearest_label,
          check_separable_data_is_classified]
