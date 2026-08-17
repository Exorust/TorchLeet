"""knowledge-distillation — soft-target loss."""
import torch
import torch.nn.functional as F

ENTRIES = ["distillation_loss"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Two terms: KL against the temperature-softened teacher, and cross-entropy against the hard labels.",
    "alpha blends them: alpha * soft + (1 - alpha) * hard.",
    "Scale the soft term by temperature**2 so its gradients stay comparable as T changes.",
]

B, C = 8, 5


def _batch(seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, C), torch.randn(B, C) * 2, torch.randint(0, C, (B,)))


def check_returns_scalar(ns):
    s, t, y = _batch()
    loss = ns.distillation_loss(s, t, y, temperature=2.0, alpha=0.5)
    loss = torch.as_tensor(loss)
    assert loss.ndim == 0, f"loss should be a scalar, got shape {tuple(loss.shape)}"
    assert float(loss) >= 0, f"loss should be non-negative, got {float(loss):.4f}"


def check_alpha_zero_is_pure_cross_entropy(ns):
    s, t, y = _batch()
    got = torch.as_tensor(ns.distillation_loss(s, t, y, temperature=2.0, alpha=0.0))
    assert torch.allclose(got.float(), F.cross_entropy(s, y), atol=1e-4), \
        (f"alpha=0 should leave only the hard-label term "
         f"({float(got):.4f} vs cross_entropy {float(F.cross_entropy(s, y)):.4f})")


def check_matching_the_teacher_lowers_the_soft_term(ns):
    _, t, y = _batch()
    same = torch.as_tensor(ns.distillation_loss(t.clone(), t, y, temperature=2.0, alpha=1.0))
    diff = torch.as_tensor(ns.distillation_loss(-t, t, y, temperature=2.0, alpha=1.0))
    assert float(same) < float(diff), \
        ("a student matching the teacher exactly must score lower than one "
         f"disagreeing with it ({float(same):.4f} vs {float(diff):.4f})")


def check_gradients_flow_to_student_only(ns):
    s, t, y = _batch()
    s.requires_grad_(True)
    t.requires_grad_(True)
    torch.as_tensor(ns.distillation_loss(s, t, y, temperature=2.0, alpha=0.5)).backward()
    assert s.grad is not None and torch.any(s.grad != 0), \
        "no gradient reached the student logits"


CHECKS = [check_returns_scalar, check_alpha_zero_is_pure_cross_entropy,
          check_matching_the_teacher_lowers_the_soft_term,
          check_gradients_flow_to_student_only]
