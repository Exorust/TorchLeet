"""custom-dataset — a torch Dataset that reads (X, y) rows out of a CSV.

The checks build their own throwaway CSV with known values, so nothing depends on
the repo's data.csv, and they never look at *how* the file is parsed — pandas,
csv, or anything else is fine. What is verified is the Dataset protocol
(`__len__`, `__getitem__`), that the values that come back are the ones in the
file, in file order, and that a DataLoader can batch it.
"""
import csv
import importlib.util
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchleet.runner import Skip

ENTRIES = ["LinearRegressionDataset"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Subclass torch.utils.data.Dataset and implement __len__ and __getitem__.",
    "Read the file once in __init__ and keep X and y as float32 tensors; "
    "__getitem__ should just index them.",
    "__getitem__(i) returns the pair (X[i], y[i]) — DataLoader stacks those into "
    "batches for you, so do not batch inside the Dataset.",
]

ROWS = [(1.0, 5.0), (2.0, 7.0), (3.0, 9.0), (4.0, 11.0), (5.0, 13.0),
        (6.0, 15.0), (7.0, 17.0), (8.0, 19.0), (9.0, 21.0), (10.0, 23.0)]


def _csv(path: Path) -> str:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["X", "y"])
        w.writerows(ROWS)
    return str(path)


ALIASES = {"pd": "pandas", "np": "numpy", "pl": "polars", "pa": "pyarrow"}


def _dataset(ns):
    """Build the solver's Dataset over a temporary CSV with known contents."""
    tmp = Path(tempfile.mkdtemp()) / "data.csv"
    path = _csv(tmp)
    try:
        return ns.LinearRegressionDataset(path)
    except ModuleNotFoundError as e:
        raise Skip(f"your Dataset needs `{e.name}` — pip install {e.name}") from None
    except NameError as e:
        # An import that failed at the top of the notebook leaves its alias
        # unbound down here. Only excuse it when the package really is absent.
        pkg = ALIASES.get(e.name, e.name)
        if importlib.util.find_spec(pkg) is None:
            raise Skip(f"your Dataset needs `{pkg}` — pip install {pkg}") from None
        raise


def check_is_a_torch_dataset(ns):
    """DataLoader only accepts something that implements the Dataset protocol."""
    cls = ns.LinearRegressionDataset
    assert isinstance(cls, type), \
        f"LinearRegressionDataset should be a class, got {type(cls).__name__}"
    assert issubclass(cls, Dataset), \
        "LinearRegressionDataset must subclass torch.utils.data.Dataset"
    for method in ("__len__", "__getitem__"):
        assert getattr(cls, method, None) is not getattr(Dataset, method, None), \
            f"LinearRegressionDataset does not implement {method}"


def check_length_is_the_row_count(ns):
    ds = _dataset(ns)
    assert len(ds) == len(ROWS), (
        f"the CSV has {len(ROWS)} data rows but len(dataset) is {len(ds)} — do not "
        "count the header row, and do not drop rows")


def check_getitem_returns_x_and_y(ns):
    ds = _dataset(ns)
    item = ds[0]
    assert isinstance(item, (tuple, list)) and len(item) == 2, (
        f"dataset[0] should be a (features, target) pair, got "
        f"{type(item).__name__}"
        + (f" of length {len(item)}" if hasattr(item, '__len__') else ""))
    x, y = item
    for name, t in (("X", x), ("y", y)):
        assert torch.is_tensor(t), \
            f"the {name} half of dataset[0] is a {type(t).__name__}, expected a tensor"
        assert t.dtype.is_floating_point, (
            f"the {name} half of dataset[0] has dtype {t.dtype}; the model needs "
            "float32, so convert when you build the tensor")


def check_values_match_the_file(ns):
    """Row i of the CSV must come back as item i — no shuffling inside Dataset."""
    ds = _dataset(ns)
    for i in (0, 3, len(ROWS) - 1):
        x, y = ds[i]
        want_x, want_y = ROWS[i]
        assert abs(float(x.reshape(-1)[0]) - want_x) < 1e-4, (
            f"dataset[{i}] X is {float(x.reshape(-1)[0])}, but row {i} of the CSV "
            f"holds {want_x} — keep the file order and pick the right column")
        assert abs(float(y.reshape(-1)[0]) - want_y) < 1e-4, (
            f"dataset[{i}] y is {float(y.reshape(-1)[0])}, but row {i} of the CSV "
            f"holds {want_y}")


def check_dataloader_batches_it(ns):
    ds = _dataset(ns)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3, (
        f"batch_size=4 over {len(ROWS)} rows should give 3 batches (4+4+2), got "
        f"{len(batches)}")
    bx, by = batches[0]
    assert bx.shape[0] == 4 and by.shape[0] == 4, (
        f"the first batch has {bx.shape[0]} X rows and {by.shape[0]} y rows, "
        "expected 4 of each")
    seen = sorted(float(v) for b in batches for v in b[0].reshape(-1))
    assert seen == sorted(r[0] for r in ROWS), (
        "iterating the DataLoader did not yield every row exactly once — got "
        f"{seen}")


CHECKS = [
    check_is_a_torch_dataset,
    check_length_is_the_row_count,
    check_getitem_returns_x_and_y,
    check_values_match_the_file,
    check_dataloader_batches_it,
]
