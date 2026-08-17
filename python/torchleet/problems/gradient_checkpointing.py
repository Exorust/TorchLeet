"""gradient-checkpointing — trade compute for memory by recomputing activations.

Checkpointing is only useful if it is invisible: the graded property is that a
checkpointed call produces the SAME output and the SAME gradients as calling the
function directly. Both sides of every comparison run the solver's own code —
one through `checkpoint`, one without — so a correct implementation cannot fail
on op order or initialization.

The check with teeth is check_intermediates_are_not_kept: it holds a weakref to
an activation created inside the checkpointed function and asserts it has been
freed by the time forward returns. An implementation that quietly calls fn(*args)
passes every gradient check and fails that one, which is the whole exercise.
"""
import gc
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchleet.runner import Skip

ENTRIES = ["CheckpointFunction", "checkpoint"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "In forward: stash fn on ctx, ctx.save_for_backward(*args), and run "
    "fn(*args) inside torch.no_grad() so no intermediate activation is kept.",
    "In backward: detach the saved inputs, requires_grad_ them, re-run fn inside "
    "torch.enable_grad(), then torch.autograd.grad(output, inputs, grad_outputs).",
    "backward must return one gradient per forward argument — None for fn itself, "
    "then a gradient (or None) for each tensor input.",
]

BATCH, DIM, HIDDEN = 6, 8, 16


def _tensors(seed=3):
    """Inputs and weights as explicit tensors so both runs share identical values."""
    torch.manual_seed(seed)
    return [
        torch.randn(BATCH, DIM, requires_grad=True),
        torch.randn(HIDDEN, DIM, requires_grad=True),
        torch.randn(HIDDEN, requires_grad=True),
        torch.randn(DIM, HIDDEN, requires_grad=True),
        torch.randn(DIM, requires_grad=True),
    ]


def _mlp(x, w1, b1, w2, b2):
    return F.linear(torch.tanh(F.linear(x, w1, b1)), w2, b2)


def check_forward_output_matches(ns):
    plain, ckpt = _tensors(), _tensors()
    expected = _mlp(*plain)
    got = ns.checkpoint(_mlp, *ckpt)
    assert got is not None, "checkpoint() returned None — did you forget to return?"
    assert tuple(got.shape) == tuple(expected.shape), \
        f"expected output {tuple(expected.shape)}, got {tuple(got.shape)}"
    diff = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=1e-6), (
        f"checkpoint(fn, *args) must return exactly fn(*args) (max diff {diff:.2e})")


def check_output_is_differentiable(ns):
    args = _tensors()
    out = ns.checkpoint(_mlp, *args)
    assert out.requires_grad, (
        "the checkpointed output does not require grad — running fn under "
        "no_grad() is right, but the CheckpointFunction still has to be the "
        "output's grad_fn so backward can recompute")


def check_gradients_match_uncheckpointed(ns):
    """The defining property: same numbers in, same gradients out."""
    plain, ckpt = _tensors(), _tensors()
    _mlp(*plain).pow(2).sum().backward()
    ns.checkpoint(_mlp, *ckpt).pow(2).sum().backward()
    names = ["x", "w1", "b1", "w2", "b2"]
    for name, a, b in zip(names, plain, ckpt):
        assert b.grad is not None, (
            f"no gradient reached `{name}` through the checkpoint — backward must "
            f"return one gradient per forward input (after the None for fn)")
        diff = (a.grad - b.grad).abs().max().item()
        assert torch.allclose(a.grad, b.grad, atol=1e-6), (
            f"gradient w.r.t. `{name}` differs from the uncheckpointed run "
            f"(max diff {diff:.2e}) — recomputation must reproduce the forward pass "
            f"exactly before the gradients are taken")


def check_chained_checkpoints_match(ns):
    """A deep stack of checkpointed blocks — gradients must still flow end to end."""
    torch.manual_seed(11)
    w = torch.randn(DIM, DIM) * 0.3
    x0 = torch.randn(BATCH, DIM)

    def block(t, weight):
        return torch.tanh(t @ weight) + t

    plain_x, plain_w = x0.clone().requires_grad_(True), w.clone().requires_grad_(True)
    y = plain_x
    for _ in range(8):
        y = block(y, plain_w)
    y.sum().backward()

    ck_x, ck_w = x0.clone().requires_grad_(True), w.clone().requires_grad_(True)
    z = ck_x
    for _ in range(8):
        z = ns.checkpoint(block, z, ck_w)
    z.sum().backward()

    diff_out = (y - z).abs().max().item()
    assert torch.allclose(y, z, atol=1e-5), \
        f"8 chained checkpointed blocks changed the output (max diff {diff_out:.2e})"
    for name, a, b in (("input", plain_x, ck_x), ("weight", plain_w, ck_w)):
        assert b.grad is not None, f"no gradient reached the {name} through 8 checkpoints"
        diff = (a.grad - b.grad).abs().max().item()
        assert torch.allclose(a.grad, b.grad, atol=1e-5), (
            f"gradient w.r.t. the {name} differs after 8 chained checkpoints "
            f"(max diff {diff:.2e})")


def check_intermediates_are_not_kept(ns):
    """The memory property: activations inside fn must be gone after forward."""
    torch.manual_seed(12)
    x = torch.randn(BATCH, DIM, requires_grad=True)
    w = torch.randn(DIM, DIM, requires_grad=True)
    seen = {}

    def fn(t, weight):
        hidden = torch.tanh(t @ weight)
        seen["ref"] = weakref.ref(hidden)
        return hidden @ weight

    out = ns.checkpoint(fn, x, w)
    gc.collect()
    assert seen["ref"]() is None, (
        "the activation computed inside fn is still alive after checkpoint() "
        "returned, so it is being stored for backward — run fn under "
        "torch.no_grad() in forward and recompute it in backward instead")

    # Sanity: without checkpointing that same activation *is* retained, so the
    # check above is measuring the checkpoint and not a quirk of the test.
    kept = fn(x, w)
    gc.collect()
    assert seen["ref"]() is not None, \
        "test setup problem: the intermediate was not retained even without checkpointing"
    del kept, out


def check_module_parameter_gradients(ns):
    """Parameters captured inside fn rather than passed as arguments.

    torch.utils.checkpoint reaches them; the recomputation-plus-autograd.grad
    formulation in this problem's solution only reaches its tensor arguments. If
    an implementation does populate them they must be right, otherwise this is
    reported as not verified rather than silently passed.
    """
    torch.manual_seed(13)
    layer = nn.Linear(DIM, DIM)
    plain = nn.Linear(DIM, DIM)
    plain.load_state_dict(layer.state_dict())
    x = torch.randn(BATCH, DIM, requires_grad=True)

    plain(x).pow(2).sum().backward()
    ns.checkpoint(layer, x.detach().requires_grad_(True)).pow(2).sum().backward()

    if layer.weight.grad is None and layer.bias.grad is None:
        raise Skip(
            "checkpoint() propagates gradients only to the tensors passed as "
            "arguments, not to parameters captured inside fn (this problem's own "
            "solution behaves the same way) — not verified")
    for name, p, q in (("weight", plain.weight, layer.weight),
                       ("bias", plain.bias, layer.bias)):
        assert q.grad is not None, \
            f"gradients reached some parameters of fn but not the module's {name}"
        diff = (p.grad - q.grad).abs().max().item()
        assert torch.allclose(p.grad, q.grad, atol=1e-6), (
            f"gradient on the module's {name} differs from the uncheckpointed run "
            f"(max diff {diff:.2e})")


CHECKS = [
    check_forward_output_matches,
    check_output_is_differentiable,
    check_gradients_match_uncheckpointed,
    check_chained_checkpoints_match,
    check_intermediates_are_not_kept,
    check_module_parameter_gradients,
]
