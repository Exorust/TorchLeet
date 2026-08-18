"""temperature-sampling."""
import torch

ENTRIES = ["temperature_sample"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Divide the logits by the temperature before the softmax.",
    "Low temperature sharpens the distribution toward the argmax; high flattens it.",
    "Sample with torch.multinomial on the resulting probabilities.",
]

V = 40


def _tok(x):
    return int(x.item()) if torch.is_tensor(x) else int(x)


def check_returns_valid_token(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    for _ in range(20):
        assert 0 <= _tok(ns.temperature_sample(logits, temperature=1.0)) < V, \
            "sampled token outside the vocabulary"


def check_low_temperature_is_greedy(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    best = int(logits.argmax())
    for _ in range(15):
        t = _tok(ns.temperature_sample(logits, temperature=0.01))
        assert t == best, \
            f"temperature 0.01 should be effectively greedy; expected {best}, got {t}"


def check_high_temperature_is_diverse(ns):
    torch.manual_seed(0)
    logits = torch.randn(V) * 5      # very peaked before temperature is applied
    seen = {_tok(ns.temperature_sample(logits, temperature=100.0)) for _ in range(200)}
    assert len(seen) > 5, \
        f"temperature 100 should flatten the distribution, but only {len(seen)} token(s) appeared"


CHECKS = [check_returns_valid_token, check_low_temperature_is_greedy,
          check_high_temperature_is_diverse]
