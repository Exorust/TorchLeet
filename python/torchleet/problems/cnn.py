"""cnn — a CNN for CIFAR-10.

A Tier C problem: the notebook ships no assertions, so these checks are authored
rather than extracted. Note EXTRAS is empty even though the notebook imports
torchvision — torchvision loads the *dataset*, while the graded unit is the model,
which is pure torch. Requiring it would skip a check that can actually run.

There is no single correct architecture here, so nothing compares against a
reference: the checks verify the contract (shape), that the network is trainable
(gradients reach every parameter, loss goes down), and that it is batch-agnostic.
"""
import torch
import torch.nn as nn

ENTRIES = ["CNNModel"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "CIFAR-10 images are 3x32x32 and there are 10 classes.",
    "Track the spatial size through each pool: 32 -> 16 after one 2x2 pool.",
    "The first Linear needs in_features = channels * height * width after flattening.",
]

SHAPE = (3, 32, 32)
CLASSES = 10


def _model(ns):
    torch.manual_seed(0)
    return ns.CNNModel()


def check_output_shape(ns):
    m = _model(ns)
    out = m(torch.randn(4, *SHAPE))
    assert out.ndim == 2, f"expected a 2D (batch, classes) output, got {out.ndim}D"
    assert out.shape[0] == 4, f"batch dim should be 4, got {out.shape[0]}"
    assert out.shape[1] == CLASSES, \
        f"CIFAR-10 has {CLASSES} classes, model outputs {out.shape[1]}"


def check_batch_agnostic(ns):
    m = _model(ns)
    for b in (1, 3, 8):
        out = m(torch.randn(b, *SHAPE))
        assert out.shape == (b, CLASSES), \
            f"batch size {b} gave {tuple(out.shape)}, expected {(b, CLASSES)}"


def check_gradients_reach_all_parameters(ns):
    m = _model(ns)
    out = m(torch.randn(2, *SHAPE))
    nn.CrossEntropyLoss()(out, torch.tensor([0, 1])).backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — these layers are not on the forward path")


def check_can_learn(ns):
    """Overfit a fixed synthetic batch; loss must fall. Catches a model that is
    wired up but cannot train (e.g. detached graph, dead activations)."""
    torch.manual_seed(0)
    m = _model(ns)
    x = torch.randn(8, *SHAPE)
    y = torch.randint(0, CLASSES, (8,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    first = lossf(m(x), y).item()
    for _ in range(30):
        opt.zero_grad()
        loss = lossf(m(x), y)
        loss.backward()
        opt.step()
    last = loss.item()
    assert last < first, (
        f"loss did not decrease over 30 steps on a fixed batch "
        f"({first:.3f} -> {last:.3f}) — the model is not learning")


CHECKS = [
    check_output_shape,
    check_batch_agnostic,
    check_gradients_reach_all_parameters,
    check_can_learn,
]
