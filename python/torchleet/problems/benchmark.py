"""benchmark — add benchmarking to a training loop.

Timing code is not gradeable; the model under test is.
"""
from torchleet.checks import common as c

ENTRIES = ["SimpleNN"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Flatten the 28x28 image before the first Linear layer.",
    "Use time.perf_counter() around the epoch, not time.time().",
    "On CUDA you must torch.cuda.synchronize() before stopping the clock.",
]
CHECKS = [
    c.output_shape("SimpleNN", (1, 28, 28), (10,)),
    c.gradients_flow("SimpleNN", (1, 28, 28)),
    c.can_learn("SimpleNN", (1, 28, 28), (10,), classification=True),
]
