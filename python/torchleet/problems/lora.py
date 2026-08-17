"""lora — low-rank adaptation.

The strongest check here is self-consistency: merging LoRA weights back into the
base layer must not change what the model outputs. That property holds for any
correct implementation regardless of how A and B are initialized or scaled
internally, so it cannot false-fail a valid solution.
"""
import torch
import torch.nn as nn

ENTRIES = ["LoRALinear", "apply_lora", "merge_lora"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "LoRA adds a low-rank detour: output = W@x + (alpha/r) * (x@A)@B.",
    "A is (d_in, rank) and B is (rank, d_out); the base layer's weights are frozen.",
    "Merging means W_new = W_old + scaling * (A@B).T — same maths, one matrix.",
]

D_IN, D_OUT, RANK = 16, 8, 4


def _model():
    torch.manual_seed(0)
    m = nn.Sequential()
    m.add_module("target", nn.Linear(D_IN, D_OUT))
    return m


def check_lora_layer_shapes_are_low_rank(ns):
    base = nn.Linear(D_IN, D_OUT)
    layer = ns.LoRALinear(base, rank=RANK, alpha=1.0)
    params = dict(layer.named_parameters())
    lows = [p for n, p in params.items() if p.requires_grad and p.dim() == 2]
    assert lows, "LoRALinear exposes no trainable 2D adapter matrices"
    for p in lows:
        assert RANK in p.shape, \
            f"trainable matrix {tuple(p.shape)} has no dimension equal to rank={RANK}"


def check_base_weights_are_frozen(ns):
    base = nn.Linear(D_IN, D_OUT)
    layer = ns.LoRALinear(base, rank=RANK, alpha=1.0)
    assert not base.weight.requires_grad, \
        "the wrapped layer's weight must be frozen (requires_grad=False)"


def check_forward_shape(ns):
    layer = ns.LoRALinear(nn.Linear(D_IN, D_OUT), rank=RANK, alpha=1.0)
    out = layer(torch.randn(3, D_IN))
    assert tuple(out.shape) == (3, D_OUT), \
        f"expected output {(3, D_OUT)}, got {tuple(out.shape)}"


def check_apply_lora_replaces_target(ns):
    m = ns.apply_lora(_model(), rank=RANK, alpha=1.0, target_modules=["target"])
    assert isinstance(m.target, ns.LoRALinear), \
        "apply_lora did not wrap the layer named in target_modules"


def check_merge_preserves_output(ns):
    """Merged and unmerged models must agree — the property that defines a merge."""
    m = ns.apply_lora(_model(), rank=RANK, alpha=2.0, target_modules=["target"])
    # Give the adapter a non-trivial value, else the merge is trivially identity.
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.dim() == 2 and RANK in p.shape:
                p.normal_(0, 0.3)
    x = torch.randn(5, D_IN)
    m.eval()
    with torch.no_grad():
        before = m(x)
        merged = ns.merge_lora(m)
        after = merged(x)
    diff = (before - after).abs().max().item()
    assert torch.allclose(before, after, atol=1e-5), (
        f"merging changed the model's output (max diff {diff:.2e}). "
        f"W_new should equal W_old + scaling * (A@B).T")


def check_merge_removes_adapter(ns):
    m = ns.apply_lora(_model(), rank=RANK, alpha=1.0, target_modules=["target"])
    merged = ns.merge_lora(m)
    assert not isinstance(merged.target, ns.LoRALinear), \
        "after merge_lora the layer should be a plain nn.Linear again"


CHECKS = [check_lora_layer_shapes_are_low_rank, check_base_weights_are_frozen,
          check_forward_shape, check_apply_lora_replaces_target,
          check_merge_preserves_output, check_merge_removes_adapter]
