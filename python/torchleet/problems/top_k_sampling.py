"""top-k-sampling."""
import torch

ENTRIES = ["top_k_sample"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Keep the k largest logits and make the rest impossible to select.",
    "Setting the discarded logits to -inf before the softmax does that cleanly.",
    "torch.topk gives you both the values and their indices.",
]

V = 60


def _tok(x):
    return int(x.item()) if torch.is_tensor(x) else int(x)


def check_returns_valid_token(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    for _ in range(20):
        t = _tok(ns.top_k_sample(logits, k=10))
        assert 0 <= t < V, f"token {t} outside [0, {V})"


def check_samples_only_from_top_k(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    for k in (1, 5, 20):
        allowed = set(torch.topk(logits, k).indices.tolist())
        for _ in range(60):
            t = _tok(ns.top_k_sample(logits, k=k))
            assert t in allowed, f"k={k}: sampled {t}, which is not among the top {k}"


def check_k_of_one_is_argmax(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    best = int(logits.argmax())
    for _ in range(15):
        assert _tok(ns.top_k_sample(logits, k=1)) == best, \
            f"k=1 must always return the argmax ({best})"


def check_larger_k_is_more_diverse(ns):
    torch.manual_seed(0)
    logits = torch.zeros(V)
    seen = {_tok(ns.top_k_sample(logits, k=V)) for _ in range(200)}
    assert len(seen) > 1, "k=vocab on a uniform distribution returned a single token"


CHECKS = [check_returns_valid_token, check_samples_only_from_top_k,
          check_k_of_one_is_argmax, check_larger_k_is_more_diverse]
