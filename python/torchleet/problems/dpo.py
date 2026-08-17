"""dpo — Direct Preference Optimization loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["get_batch_logps", "dpo_loss"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "get_batch_logps sums the log-probability of each actual next token in the sequence.",
    "Shift by one: logits at position t predict the token at position t+1.",
    "DPO = -logsigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))).",
]

B, S, VOCAB = 4, 6, 20


class _UniformLM(nn.Module):
    """Every token equally likely, so the expected log-prob is computable."""

    def __init__(self, vocab=VOCAB):
        super().__init__()
        self.vocab = vocab
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        b, s = input_ids.shape
        return torch.zeros(b, s, self.vocab) + self.dummy


def check_logps_shape(ns):
    ids = torch.randint(0, VOCAB, (B, S))
    out = ns.get_batch_logps(_UniformLM(), ids)
    assert tuple(out.shape) == (B,), \
        f"expected one log-prob per sequence {(B,)}, got {tuple(out.shape)}"


def check_logps_are_negative(ns):
    out = ns.get_batch_logps(_UniformLM(), torch.randint(0, VOCAB, (B, S)))
    assert bool((out < 0).all()), \
        f"log probabilities must be negative, got {out.tolist()}"


def check_logps_match_uniform_expectation(ns):
    """With uniform logits every predicted token has probability 1/VOCAB."""
    out = ns.get_batch_logps(_UniformLM(), torch.randint(0, VOCAB, (B, S)))
    import math
    per_token = math.log(1.0 / VOCAB)
    assert torch.allclose(out, torch.full((B,), per_token * (S - 1)), atol=1e-3), (
        f"under a uniform model each of the {S - 1} predicted tokens contributes "
        f"log(1/{VOCAB}); expected {per_token * (S - 1):.3f}, got {out[0]:.3f}")


def _logps(seed=0):
    torch.manual_seed(seed)
    return [torch.randn(B) for _ in range(4)]


def check_loss_is_scalar_and_positive(ns):
    pc, pr, rc, rr = _logps()
    loss = torch.as_tensor(ns.dpo_loss(pc, pr, rc, rr, beta=0.1))
    assert loss.ndim == 0, f"DPO loss should be a scalar, got shape {tuple(loss.shape)}"
    assert float(loss) > 0, f"-logsigmoid(...) is always positive, got {float(loss):.4f}"


def check_matches_closed_form(ns):
    pc, pr, rc, rr = _logps(1)
    beta = 0.2
    expected = -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()
    got = torch.as_tensor(ns.dpo_loss(pc, pr, rc, rr, beta=beta))
    assert torch.allclose(got.float(), expected, atol=1e-5), \
        f"disagrees with the DPO closed form ({float(got):.5f} vs {float(expected):.5f})"


def check_preferring_chosen_lowers_loss(ns):
    """The behaviour the loss exists to produce."""
    base = torch.zeros(B)
    good = torch.as_tensor(ns.dpo_loss(base + 2.0, base - 2.0, base, base, beta=0.1))
    bad = torch.as_tensor(ns.dpo_loss(base - 2.0, base + 2.0, base, base, beta=0.1))
    assert float(good) < float(bad), (
        "a policy that upweights the chosen response must score a LOWER loss "
        f"than one that upweights the rejected ({float(good):.4f} vs {float(bad):.4f})")


CHECKS = [check_logps_shape, check_logps_are_negative,
          check_logps_match_uniform_expectation, check_loss_is_scalar_and_positive,
          check_matches_closed_form, check_preferring_chosen_lowers_loss]
