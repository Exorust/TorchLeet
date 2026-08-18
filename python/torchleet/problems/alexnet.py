"""alexnet — AlexNet from scratch."""
from torchleet.checks import common as c

ENTRIES = ["AlexNet"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "AlexNet expects 3x224x224 input and ends in a classifier over num_classes.",
    "Five conv layers with max-pooling, then three fully connected layers.",
    "Track the spatial size through each stride and pool to size the first Linear.",
]
CHECKS = [
    c.output_shape("AlexNet", (3, 224, 224), (10,), batch=2),
    c.batch_agnostic("AlexNet", (3, 224, 224), (10,), sizes=(1, 2)),
    c.gradients_flow("AlexNet", (3, 224, 224)),
]
