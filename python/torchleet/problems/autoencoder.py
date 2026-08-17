"""autoencoder — convolutional autoencoder for anomaly detection."""
import torch
from torchleet.checks import common as c

ENTRIES = ["Autoencoder"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "The decoder must undo the encoder's downsampling, so the output matches the input shape.",
    "ConvTranspose2d with stride=2 doubles the spatial size; output_padding fixes off-by-ones.",
    "Train it to reconstruct its own input with MSELoss.",
]

SHAPE = (1, 28, 28)


def check_reconstructs_input_shape(ns):
    """An autoencoder whose output shape differs from its input cannot reconstruct."""
    m = c.build(ns, "Autoencoder")
    out = m(torch.randn(4, *SHAPE))
    assert tuple(out.shape) == (4, *SHAPE), \
        f"output {tuple(out.shape)} does not match the input {(4, *SHAPE)}"


def check_encoder_downsamples(ns):
    """The encoder must reduce the spatial resolution.

    Note this checks height/width, not total element count: a convolutional
    autoencoder legitimately grows the channel dimension while shrinking space,
    so its latent can hold more numbers than the input and still be correct.
    """
    m = c.build(ns, "Autoencoder")
    enc = getattr(m, "encoder", None)
    if enc is None:
        from torchleet.runner import Skip
        raise Skip("no .encoder attribute to inspect")
    z = enc(torch.randn(1, *SHAPE))
    assert z.shape[-1] < SHAPE[-1] and z.shape[-2] < SHAPE[-2], (
        f"encoder output is {tuple(z.shape[-2:])} but the input is {SHAPE[-2:]} — "
        "no spatial downsampling")


CHECKS = [
    check_reconstructs_input_shape,
    check_encoder_downsamples,
    c.gradients_flow("Autoencoder", SHAPE),
    c.can_learn("Autoencoder", SHAPE, SHAPE, batch=4, steps=30),
]
