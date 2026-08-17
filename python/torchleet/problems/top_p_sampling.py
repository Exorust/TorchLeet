"""top-p-sampling — nucleus sampling."""
import torch

ENTRIES = ["top_p_sample"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Sort probabilities descending, then take the smallest prefix whose cumulative mass >= p.",
    "Always keep at least one token, or a peaked distribution can select nothing.",
    "Renormalize the surviving probabilities before sampling from them.",
]

V = 50


def _nucleus(logits, p, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    s, idx = probs.sort(descending=True)
    keep = (s.cumsum(-1) - s) < p          # prefix strictly before mass p is exceeded
    keep[0] = True                          # never drop the top token
    return set(idx[keep].tolist())


def check_returns_valid_token(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    for _ in range(20):
        t = ns.top_p_sample(logits, p=0.9)
        t = int(t.item()) if torch.is_tensor(t) else int(t)
        assert 0 <= t < V, f"token {t} outside [0, {V})"


def check_samples_only_from_nucleus(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    allowed = _nucleus(logits, 0.5)
    for _ in range(80):
        t = ns.top_p_sample(logits, p=0.5)
        t = int(t.item()) if torch.is_tensor(t) else int(t)
        assert t in allowed, \
            f"sampled token {t} is outside the top-p=0.5 nucleus {sorted(allowed)}"


def check_tiny_p_gives_argmax(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    best = int(logits.argmax())
    for _ in range(15):
        t = ns.top_p_sample(logits, p=0.01)
        t = int(t.item()) if torch.is_tensor(t) else int(t)
        assert t == best, \
            f"with p tiny only the top token survives; expected {best}, got {t}"


def check_large_p_can_reach_beyond_top_token(ns):
    torch.manual_seed(0)
    logits = torch.zeros(V)      # uniform: every token is plausible
    seen = set()
    for _ in range(150):
        t = ns.top_p_sample(logits, p=1.0)
        seen.add(int(t.item()) if torch.is_tensor(t) else int(t))
    assert len(seen) > 1, \
        "p=1.0 on a uniform distribution always returned one token — is it sampling?"


CHECKS = [check_returns_valid_token, check_samples_only_from_nucleus,
          check_tiny_p_gives_argmax, check_large_p_can_reach_beyond_top_token]
