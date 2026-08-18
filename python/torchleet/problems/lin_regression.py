"""lin-regression — linear regression from scratch."""
from torchleet.checks import common as c

ENTRIES = ["LinearRegressionModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "A single nn.Linear(1, 1) is the whole model.",
    "forward() just applies that layer to the input.",
    "Train with MSELoss and an SGD/Adam optimizer.",
]
CHECKS = [
    c.output_shape("LinearRegressionModel", (1,), (1,)),
    c.batch_agnostic("LinearRegressionModel", (1,), (1,)),
    c.gradients_flow("LinearRegressionModel", (1,)),
    c.can_learn("LinearRegressionModel", (1,), (1,)),
]
