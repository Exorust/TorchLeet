"""vit-mae — patch embedding, a ViT encoder, and masked-autoencoder pretraining.

Graded by properties rather than by diffing against a reference ViT:

* patch embedding is checked by *causal influence* — poke one 8x8 block of the
  image and exactly one output token may move, and the block -> token map must
  be a bijection. That is what "non-overlapping patches" means, and it holds for
  a Conv2d implementation and an unfold/reshape one alike.
* the ViT's CLS output must be the first token of its own return_all_tokens
  output (self-consistency), must see every patch, and must *not* be invariant
  to shuffling patches — without positional embeddings a transformer is
  permutation equivariant, so that check is exactly the positional-embedding test.
* the MAE loss is pinned in closed form by feeding a per-sample constant image:
  then the reconstruction target is a known number regardless of how the solver
  orders pixels inside a patch, and the masked-only average can be recomputed
  from the returned pred and mask.
"""
import torch

ENTRIES = ["PatchEmbedding", "ViT", "MAE"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "PatchEmbedding is a Conv2d with kernel_size = stride = patch_size, then "
    "flatten(2).transpose(1, 2) to get (B, n_patches, embed_dim).",
    "ViT: prepend a learned [CLS] token, add a positional embedding of length "
    "n_patches + 1, run the encoder, and return either the CLS row or every token.",
    "MAE: shuffle patch indices with argsort(rand), keep the first "
    "int(N*(1-mask_ratio)), encode only those, re-insert mask tokens, unshuffle "
    "with ids_restore, and average the squared error over MASKED patches only "
    "-- (per_patch_mse * mask).sum() / mask.sum().",
]

IMG, PATCH, CH = 32, 8, 3
GRID = IMG // PATCH
NP = GRID * GRID
EMB = 32
PATCH_DIM = PATCH * PATCH * CH


def _build(cls, *kwargsets):
    last = None
    for kw in kwargsets:
        try:
            return cls(**kw)
        except TypeError as e:
            last = e
    raise AssertionError(
        f"could not construct {getattr(cls, '__name__', cls)} with the documented "
        f"keyword arguments ({last})")


def _patch_embed(ns):
    return _build(ns.PatchEmbedding,
                  dict(img_size=IMG, patch_size=PATCH, in_channels=CH, embed_dim=EMB),
                  dict(img_size=IMG, patch_size=PATCH, embed_dim=EMB),
                  dict())


def _vit(ns):
    m = _build(ns.ViT,
               dict(img_size=IMG, patch_size=PATCH, in_channels=CH, embed_dim=EMB,
                    n_heads=2, n_layers=1),
               dict(img_size=IMG, patch_size=PATCH, in_channels=CH, embed_dim=EMB),
               dict())
    m.eval()          # the encoder layers carry dropout; sampling must be stable
    return m


def _mae(ns, mask_ratio=0.75):
    m = _build(ns.MAE,
               dict(img_size=IMG, patch_size=PATCH, in_channels=CH,
                    encoder_embed_dim=EMB, encoder_heads=2, encoder_layers=1,
                    decoder_embed_dim=16, decoder_heads=2, decoder_layers=1,
                    mask_ratio=mask_ratio),
               dict(img_size=IMG, patch_size=PATCH, in_channels=CH,
                    mask_ratio=mask_ratio),
               dict(mask_ratio=mask_ratio))
    m.eval()
    return m


def _mae_forward(m, images):
    out = m(images)
    assert isinstance(out, (tuple, list)) and len(out) == 3, \
        "MAE.forward should return (loss, pred, mask)"
    loss, pred, mask = out
    return torch.as_tensor(loss), pred, mask


# --------------------------------------------------------------------------
# PatchEmbedding
# --------------------------------------------------------------------------
def check_patch_embedding_shape(ns):
    pe = _patch_embed(ns)
    out = pe(torch.randn(2, CH, IMG, IMG))
    assert tuple(out.shape) == (2, NP, EMB), (
        f"a {IMG}x{IMG} image cut into {PATCH}x{PATCH} patches gives {NP} tokens, so "
        f"the output should be {(2, NP, EMB)}; got {tuple(out.shape)}")
    assert hasattr(pe, "n_patches") and int(pe.n_patches) == NP, (
        f"PatchEmbedding should expose n_patches = (img_size // patch_size)**2 = "
        f"{NP} (the ViT needs it to size its positional embedding); got "
        f"{getattr(pe, 'n_patches', 'no attribute')}")


def check_each_token_comes_from_exactly_one_patch(ns):
    """Non-overlapping patches: one block in, one token out, and it's a bijection."""
    pe = _patch_embed(ns)
    pe.eval()
    x = torch.zeros(1, CH, IMG, IMG)
    with torch.no_grad():
        base = pe(x)
        hits = []
        for r in range(GRID):
            for c in range(GRID):
                x2 = x.clone()
                x2[0, :, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 3.0
                moved = ((pe(x2) - base).abs().sum(-1)[0] > 1e-6).nonzero().flatten()
                assert moved.numel() == 1, (
                    f"changing only the pixels of patch (row {r}, col {c}) moved "
                    f"{moved.numel()} output tokens — patches must be "
                    f"non-overlapping, so exactly one token may depend on it")
                hits.append(int(moved[0]))
    assert sorted(hits) == list(range(NP)), (
        f"the patch -> token map must be a bijection over the {NP} tokens; "
        f"patches landed on tokens {hits}")
    assert hits[0] == 0 and hits[-1] == NP - 1, (
        f"the top-left patch must be the first token and the bottom-right patch "
        f"the last; they landed on tokens {hits[0]} and {hits[-1]}")


def check_patch_embedding_gradients_flow(ns):
    pe = _patch_embed(ns)
    x = torch.randn(2, CH, IMG, IMG, requires_grad=True)
    pe(x).sum().backward()
    assert x.grad is not None and bool((x.grad != 0).any()), \
        "no gradient reached the image — the projection is detached"


# --------------------------------------------------------------------------
# ViT
# --------------------------------------------------------------------------
def check_vit_output_shapes(ns):
    vit = _vit(ns)
    x = torch.randn(2, CH, IMG, IMG)
    with torch.no_grad():
        cls = vit(x)
        allt = vit(x, return_all_tokens=True)
    assert tuple(cls.shape) == (2, EMB), (
        f"the default forward returns the pooled [CLS] representation {(2, EMB)}, "
        f"got {tuple(cls.shape)}")
    assert tuple(allt.shape) == (2, NP + 1, EMB), (
        f"return_all_tokens should give the [CLS] token plus all {NP} patch tokens, "
        f"i.e. {(2, NP + 1, EMB)}; got {tuple(allt.shape)}")


def check_vit_cls_is_the_first_token(ns):
    """Self-consistency between the two return modes."""
    vit = _vit(ns)
    x = torch.randn(2, CH, IMG, IMG)
    with torch.no_grad():
        cls = vit(x)
        allt = vit(x, return_all_tokens=True)
    assert torch.allclose(cls, allt[:, 0], atol=1e-5), (
        "the pooled output must be row 0 of the full token output — the [CLS] "
        "token is prepended, so it sits at index 0; max deviation "
        f"{float((cls - allt[:, 0]).abs().max()):.6f}")


def check_vit_cls_attends_to_every_patch(ns):
    vit = _vit(ns)
    x = torch.zeros(1, CH, IMG, IMG)
    with torch.no_grad():
        base = vit(x)
        for r in range(GRID):
            for c in range(GRID):
                x2 = x.clone()
                x2[0, :, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 2.0
                assert not torch.allclose(vit(x2), base, atol=1e-6), (
                    f"changing patch (row {r}, col {c}) left the [CLS] output "
                    f"untouched — self-attention should let CLS see every patch")


def check_vit_is_position_aware(ns):
    """No positional embedding => a transformer is permutation invariant at CLS."""
    vit = _vit(ns)
    torch.manual_seed(0)
    x = torch.randn(1, CH, IMG, IMG)
    x2 = x.clone()
    a = x[:, :, 0:PATCH, 0:PATCH].clone()
    b = x[:, :, PATCH:2 * PATCH, PATCH:2 * PATCH].clone()
    x2[:, :, 0:PATCH, 0:PATCH] = b
    x2[:, :, PATCH:2 * PATCH, PATCH:2 * PATCH] = a
    with torch.no_grad():
        assert not torch.allclose(vit(x), vit(x2), atol=1e-6), (
            "swapping two patches inside the image did not change the [CLS] "
            "output. A transformer encoder is permutation equivariant, so this "
            "means no positional embedding is being added to the patch tokens.")


def check_vit_gradients_flow(ns):
    vit = _vit(ns)
    x = torch.randn(2, CH, IMG, IMG, requires_grad=True)
    out = vit(x)
    # a plain .sum() has (mathematically) zero gradient through the final
    # LayerNorm, so weight the output randomly before reducing
    torch.manual_seed(0)
    (out * torch.randn_like(out)).sum().backward()
    assert x.grad is not None and bool((x.grad != 0).any()), \
        "no gradient reached the image through the ViT"
    grads = [p.grad for p in vit.parameters() if p.grad is not None]
    assert grads and any(bool((g != 0).any()) for g in grads), \
        "no ViT parameter received a gradient"


# --------------------------------------------------------------------------
# MAE
# --------------------------------------------------------------------------
def check_mae_output_shapes(ns):
    mae = _mae(ns)
    loss, pred, mask = _mae_forward(mae, torch.randn(2, CH, IMG, IMG))
    assert loss.ndim == 0, f"the reconstruction loss should be a scalar, got shape {tuple(loss.shape)}"
    assert float(loss) >= 0 and torch.isfinite(loss), \
        f"the loss is a mean squared error: it must be finite and non-negative, got {float(loss)}"
    assert tuple(pred.shape) == (2, NP, PATCH_DIM), (
        f"the decoder predicts pixel values for every patch, so pred should be "
        f"(B, n_patches, patch_size**2 * in_channels) = {(2, NP, PATCH_DIM)}; got "
        f"{tuple(pred.shape)}")
    assert tuple(mask.shape) == (2, NP), \
        f"mask should mark each patch of each image {(2, NP)}, got {tuple(mask.shape)}"


def check_mask_marks_the_masked_fraction(ns):
    for ratio in (0.5, 0.75):
        mae = _mae(ns, mask_ratio=ratio)
        _, _, mask = _mae_forward(mae, torch.randn(4, CH, IMG, IMG))
        m = mask.float()
        assert bool(((m == 0) | (m == 1)).all()), \
            f"mask must be binary (1 = masked, 0 = visible); got values {sorted(set(m.flatten().tolist()))}"
        per_row = m.sum(dim=1)
        want = round(NP * ratio)
        assert bool((per_row - want).abs().max() <= 1), (
            f"with mask_ratio={ratio} and {NP} patches, about {want} patches per "
            f"image should be masked (mask==1); got {per_row.tolist()}. If these "
            f"are the visible counts, the mask convention is inverted.")


def check_zero_mask_ratio_masks_nothing(ns):
    mae = _mae(ns, mask_ratio=0.0)
    _, _, mask = _mae_forward(mae, torch.randn(2, CH, IMG, IMG))
    assert float(mask.float().sum()) == 0.0, (
        "with mask_ratio=0 every patch is kept, so the mask must be all zeros; "
        f"got {int(mask.float().sum())} masked patches out of {2 * NP}")


def check_masking_is_random_per_forward(ns):
    mae = _mae(ns)
    imgs = torch.randn(2, CH, IMG, IMG)
    masks = [_mae_forward(mae, imgs)[2] for _ in range(4)]
    assert any(not torch.equal(masks[0], m) for m in masks[1:]), (
        "four forward passes produced the same mask — MAE must sample a fresh "
        "random subset of patches each time (argsort of torch.rand), not a fixed one")


def check_loss_averages_over_masked_patches_only(ns):
    """A per-sample constant image makes the target a known number, whatever
    order the solver flattens pixels inside a patch."""
    mae = _mae(ns)
    consts = torch.tensor([0.25, -0.5, 1.5]).reshape(3, 1, 1, 1)
    imgs = consts.expand(3, CH, IMG, IMG).clone()
    loss, pred, mask = _mae_forward(mae, imgs)
    m = mask.float()
    per_patch = ((pred - consts.reshape(3, 1, 1)) ** 2).mean(dim=-1)   # (B, N)
    expected = float((per_patch * m).sum() / m.sum())
    unmasked = float(per_patch.mean())
    assert abs(float(loss) - expected) < 1e-4, (
        f"the MAE objective is the squared error against the original pixels, "
        f"averaged over MASKED patches only: (per_patch_mse * mask).sum() / "
        f"mask.sum() = {expected:.6f}; got {float(loss):.6f} (averaging over all "
        f"patches would give {unmasked:.6f})")


def check_mae_gradients_flow(ns):
    mae = _mae(ns)
    loss, _, _ = _mae_forward(mae, torch.randn(2, CH, IMG, IMG))
    loss.backward()
    named = [(n, p.grad) for n, p in mae.named_parameters()]
    assert any(g is not None and bool((g != 0).any()) for _, g in named), \
        "no MAE parameter received a gradient from the reconstruction loss"
    dec = [(n, g) for n, g in named
           if g is not None and "decod" in n.lower() and bool((g != 0).any())]
    enc = [(n, g) for n, g in named
           if g is not None and ("encod" in n.lower() or "patch" in n.lower())
           and bool((g != 0).any())]
    assert dec, "the decoder received no gradient — check that pred comes from it"
    assert enc, (
        "the encoder / patch embedding received no gradient — the visible patches "
        "must flow through the encoder into the decoder, not be detached")


CHECKS = [
    check_patch_embedding_shape,
    check_each_token_comes_from_exactly_one_patch,
    check_patch_embedding_gradients_flow,
    check_vit_output_shapes,
    check_vit_cls_is_the_first_token,
    check_vit_cls_attends_to_every_patch,
    check_vit_is_position_aware,
    check_vit_gradients_flow,
    check_mae_output_shapes,
    check_mask_marks_the_masked_fraction,
    check_zero_mask_ratio_masks_nothing,
    check_masking_is_random_per_forward,
    check_loss_averages_over_masked_patches_only,
    check_mae_gradients_flow,
]
