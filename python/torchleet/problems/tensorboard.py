"""tensorboard — visualize training with TensorBoard.

The model is graded; whether SummaryWriter was called is not something a grader
can meaningfully verify, so it is left to the notebook.
"""
from torchleet.checks import common as c

ENTRIES = ["LinearRegressionModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "The model itself is a plain nn.Linear(1, 1).",
    "Log with writer.add_scalar('Loss/train', loss.item(), step).",
    "Run `tensorboard --logdir=runs` to view it.",
]
CHECKS = [
    c.output_shape("LinearRegressionModel", (1,), (1,)),
    c.gradients_flow("LinearRegressionModel", (1,)),
    c.can_learn("LinearRegressionModel", (1,), (1,)),
]
