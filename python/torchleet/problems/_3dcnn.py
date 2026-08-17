"""3dcnn — a 3D encoder-decoder over CT volumes, plus the Dice objective.

MedCNN is handed a stand-in backbone built from plain torch that behaves like a
headless ResNet-18 (3-channel images in, 512 feature maps at 1/32 resolution
out), so no torchvision download is needed to grade it. The checks are structural
— a segmentation head must return a per-voxel probability map at the *input*
resolution, and the borrowed backbone must actually be on the gradient path.

compute_dice_loss is pinned against the Dice formula itself on binary masks, and
both common conventions (the coefficient, or 1 - coefficient) are accepted since
the problem does not say which one to return.
"""
import torch
import torch.nn as nn

ENTRIES = ["MedCNN", "compute_dice_loss"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "The input is [B, D, C, W, H]. Fold depth into the batch (view(B*D, C, W, H)) "
    "for the 2D backbone, then unfold and permute to [B, C, D, W, H] for Conv3d.",
    "ResNet-18 with its last two children stripped divides W and H by 32, so the "
    "ConvTranspose3d stack has to multiply them back by 32 — use kernel/stride "
    "(1, k, k) so the depth axis is left alone.",
    "Dice over binary masks is 2*|pred ∩ label| / (|pred| + |label|); finish the "
    "network with sigmoid so pred is a probability the overlap can be computed on.",
]

BATCH, DEPTH, CH, SIZE = 1, 2, 3, 64
DOWNSAMPLE = 32          # what a headless ResNet-18 does to W and H
FEATURES = 512           # ...and how many channels it emits


class _Backbone(nn.Module):
    """Stands in for `nn.Sequential(*list(resnet18().children())[:-2])`."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(CH, 8, 3, padding=1)
        self.pool = nn.AvgPool2d(DOWNSAMPLE)
        self.proj = nn.Conv2d(8, FEATURES, 1)

    def forward(self, x):
        return self.proj(self.pool(torch.relu(self.stem(x))))


def _model(ns, seed=0):
    torch.manual_seed(seed)
    return ns.MedCNN(backbone=_Backbone())


def _volume(batch=BATCH, depth=DEPTH, size=SIZE):
    return torch.randn(batch, depth, CH, size, size)


def check_segmentation_output_shape(ns):
    """One probability per voxel, at the resolution of the input volume."""
    m = _model(ns)
    m.eval()
    x = _volume()
    out = m(x)
    assert out.ndim == 5, (
        f"expected a 5D volume out of a 5D volume, got {out.ndim}D "
        f"{tuple(out.shape)}")
    assert out.shape[0] == BATCH, \
        f"batch dim should be {BATCH}, got {out.shape[0]}"
    assert tuple(out.shape[-2:]) == (SIZE, SIZE), (
        f"the mask is {tuple(out.shape[-2:])} but the slices are {(SIZE, SIZE)} — "
        f"the backbone divides W and H by {DOWNSAMPLE}, so the transposed "
        "convolutions have to put that back")
    assert out.numel() == BATCH * DEPTH * SIZE * SIZE, (
        f"expected one channel of {DEPTH} slices per volume "
        f"({BATCH * DEPTH * SIZE * SIZE} values), got {out.numel()} in "
        f"{tuple(out.shape)} — the final layer should map down to out_channel=1 "
        "and keep all the slices")


def check_output_is_a_probability_map(ns):
    m = _model(ns)
    m.eval()
    with torch.no_grad():
        out = m(_volume())
    lo, hi = float(out.min()), float(out.max())
    assert lo >= -1e-5 and hi <= 1.0 + 1e-5, (
        f"predictions span [{lo:.3f}, {hi:.3f}] — a segmentation head returns a "
        "per-voxel probability, so squash the final conv with sigmoid before the "
        "Dice loss sees it")


def check_backbone_is_trained_too(ns):
    """Transfer learning is the point: the borrowed net must be on the path."""
    m = _model(ns)
    m(_volume()).sum().backward()
    backbone = [(n, p) for n, p in m.named_parameters() if n.startswith("backbone")]
    assert backbone, "the backbone was not registered as a submodule"
    dead = [n for n, p in backbone
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached the backbone ({', '.join(dead[:3])}) — its features "
        "are not feeding the 3D stack")


def check_gradients_reach_all_parameters(ns):
    m = _model(ns)
    m(_volume()).sum().backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — these layers are not on the forward path")


def check_handles_a_different_depth(ns):
    """Depth is a data dimension, not a constant baked into the model."""
    m = _model(ns)
    m.eval()
    for depth in (1, 3):
        out = m(_volume(depth=depth))
        assert out.numel() == BATCH * depth * SIZE * SIZE, (
            f"a volume of {depth} slices produced {tuple(out.shape)} "
            f"({out.numel()} values), expected {BATCH * depth * SIZE * SIZE} — "
            "the 3D convolutions should keep the depth axis intact")


# ------------------------------------------------------------------ dice


def _masks(pattern_pred, pattern_label, batch=2, depth=3):
    """Tile one slice pattern over the whole volume.

    Every (batch, slice) then has identical overlap, so a per-slice mean and a
    global sum give the same number and both reductions are accepted.
    """
    p = torch.tensor(pattern_pred, dtype=torch.float32).view(1, 1, 1, 4, 4)
    l = torch.tensor(pattern_label, dtype=torch.float32).view(1, 1, 1, 4, 4)
    rep = (batch, depth, 1, 1, 1)
    return p.repeat(rep).contiguous(), l.repeat(rep).contiguous()


# 16 voxels per slice: 8 predicted, 8 labelled, 6 of them shared.
_PRED = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
_LABEL = [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
_DICE = 2 * 6 / (8 + 8)          # = 0.75


def check_dice_returns_a_scalar(ns):
    p, l = _masks(_PRED, _LABEL)
    got = torch.as_tensor(ns.compute_dice_loss(p, l))
    assert got.numel() == 1, (
        f"compute_dice_loss should reduce the whole volume to one number, got a "
        f"tensor of shape {tuple(got.shape)} — the loss you call .backward() on "
        "has to be a scalar")


def check_dice_matches_the_formula(ns):
    """Pinned on binary masks, where every soft-Dice variant agrees."""
    p, l = _masks(_PRED, _LABEL)
    got = float(torch.as_tensor(ns.compute_dice_loss(p, l)))
    assert min(abs(got - _DICE), abs(got - (1 - _DICE))) < 1e-3, (
        f"6 of 8 predicted voxels overlap 8 labelled ones, so the Dice "
        f"coefficient 2*6/(8+8) is {_DICE:.2f} and the Dice loss is "
        f"{1 - _DICE:.2f} — got {got:.4f}")


def check_dice_is_symmetric(ns):
    p, l = _masks(_PRED, _LABEL)
    a = float(torch.as_tensor(ns.compute_dice_loss(p, l)))
    b = float(torch.as_tensor(ns.compute_dice_loss(l, p)))
    assert abs(a - b) < 1e-5, (
        f"swapping prediction and label changed the value ({a:.4f} vs {b:.4f}); "
        "Dice is symmetric in the two masks")


def check_dice_separates_perfect_from_disjoint(ns):
    """A perfect mask and its complement must land at opposite ends."""
    p, l = _masks(_LABEL, _LABEL)
    perfect = float(torch.as_tensor(ns.compute_dice_loss(p, l)))
    inv, l2 = _masks([1 - v for v in _LABEL], _LABEL)
    disjoint = float(torch.as_tensor(ns.compute_dice_loss(inv, l2)))
    for name, v in (("perfect overlap", perfect), ("no overlap", disjoint)):
        assert -1e-4 <= v <= 1.0 + 1e-4, \
            f"{name} gave {v:.4f}; Dice stays in [0, 1] for masks in [0, 1]"
    assert abs(perfect - disjoint) > 0.9, (
        f"a perfect prediction scores {perfect:.4f} and a completely wrong one "
        f"{disjoint:.4f} — those should be 1 and 0 (or 0 and 1); the overlap term "
        "is not driving the result")


def check_dice_is_differentiable(ns):
    p, l = _masks(_PRED, _LABEL)
    p = p.clone().requires_grad_(True)
    torch.as_tensor(ns.compute_dice_loss(p, l)).backward()
    assert p.grad is not None and torch.any(p.grad != 0), (
        "no gradient reached the prediction — keep the soft (non-thresholded) "
        "probabilities in the formula, since rounding kills the gradient")


CHECKS = [
    check_segmentation_output_shape,
    check_output_is_a_probability_map,
    check_backbone_is_trained_too,
    check_gradients_reach_all_parameters,
    check_handles_a_different_depth,
    check_dice_returns_a_scalar,
    check_dice_matches_the_formula,
    check_dice_is_symmetric,
    check_dice_separates_perfect_from_disjoint,
    check_dice_is_differentiable,
]
