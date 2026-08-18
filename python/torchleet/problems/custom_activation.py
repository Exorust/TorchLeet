"""custom-activation — define and use a custom activation function."""
import torch
from torchleet.checks import common as c

ENTRIES = ["CustomActivationModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "An activation is just a function of the tensor; you can write it inline or as an nn.Module.",
    "Make sure it is differentiable — use torch ops, not Python branching on tensor values.",
    "Apply it between the linear layers in forward().",
]


def check_activation_is_nonlinear(ns):
    """A model whose activation is the identity is not doing anything."""
    m = c.build(ns, "CustomActivationModel")
    x = torch.linspace(-3, 3, 50).unsqueeze(1)
    with torch.no_grad():
        y = m(x)
    slopes = (y[1:] - y[:-1]).squeeze()
    assert float(slopes.std()) > 1e-6, \
        "output is affine in the input — the activation appears to be the identity"


CHECKS = [
    c.output_shape("CustomActivationModel", (1,), (1,)),
    c.batch_agnostic("CustomActivationModel", (1,), (1,)),
    c.gradients_flow("CustomActivationModel", (1,)),
    check_activation_is_nonlinear,
    c.can_learn("CustomActivationModel", (1,), (1,)),
]
