"""custom-dnn — a deep neural network from scratch."""
from torchleet.checks import common as c

ENTRIES = ["DNNModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Stack Linear layers with a nonlinearity between them.",
    "Without an activation between layers the whole stack collapses to one linear map.",
    "The last layer's out_features is your output dimension.",
]
CHECKS = [
    c.output_shape("DNNModel", (2,), (1,)),
    c.batch_agnostic("DNNModel", (2,), (1,)),
    c.gradients_flow("DNNModel", (2,)),
    c.can_learn("DNNModel", (2,), (1,)),
]
