"""swin-transformer — window partition/reverse, windowed attention, the
shifted-window mask, and the block (plain and with a relative position bias).

Everything is checked as a property:

* partition/reverse are graded by the exact round trip window_reverse(
  window_partition(x)) == x, plus where the first windows land.
* window_attention is graded against torch's own softmax/matmul, and by feeding
  one-hot values so the attention weights themselves become readable: they must
  sum to 1 and be exactly 0 wherever the shifted-window mask says -inf.
* the mask is graded as geometry: an equivalence relation per window, symmetric,
  no dead rows, and per-window allowed counts of 16 / 8 / 8 / 4 for an 8x8 map.
* the blocks are graded by their receptive field. Poking a single token may only
  move the tokens it is allowed to attend to -- the whole 4x4 window for W-MSA,
  and just the 2x2 corner that survives the cyclic shift for SW-MSA. Window
  attention that quietly became global attention fails this.
* the relative position bias is graded by softmax shift invariance: a *constant*
  bias table must leave the block's output untouched while a random one must not.

Sizes match the notebook's worked configuration (8x8 map, 4x4 windows, dim 32)
because the reference window_attention closes over those module-level values.
"""
import math

import torch

ENTRIES = ["window_partition", "window_reverse", "window_attention",
           "create_shifted_window_mask", "SwinBlock", "relative_position_index",
           "SwinBlockRelPosBias"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "window_partition: reshape (B, H, W, C) to "
    "(B, H/ws, ws, W/ws, ws, C), permute the two window axes next to each other, "
    "then flatten. window_reverse undoes exactly that permutation.",
    "The shifted mask compares region labels: label each pre-shift position by "
    "which of the (up-to) 4 wrapped regions it belongs to, roll the label map the "
    "same way you roll the features, partition it, and allow a pair only when the "
    "two labels match.",
    "The relative position bias is one learned scalar per relative offset: build a "
    "(w*w, w*w) index map from (row_i - row_j, col_i - col_j) shifted to be "
    "non-negative, gather the table with it, and ADD the result to the attention "
    "scores before the softmax.",
]

B, H, W, C = 2, 8, 8, 32
WS, SHIFT = 4, 2
NW = (H // WS) * (W // WS)          # 4 windows
NTOK = WS * WS                      # 16 tokens per window
TABLE = (2 * WS - 1) ** 2           # 49 distinct relative offsets


def _x(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, H, W, C, generator=g)


def _qkv(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B * NW, NTOK, C, generator=g) for _ in range(3)]


def _onehot_values():
    """v whose first NTOK channels are the identity, so the attention output
    reads back the attention weights directly."""
    v = torch.zeros(B * NW, NTOK, C)
    v[:, :, :NTOK] = torch.eye(NTOK)
    return v


def _block(ns, cls_name, shift, seed=0):
    torch.manual_seed(seed)
    blk = getattr(ns, cls_name)(C, WS, shift, H)
    blk.eval()
    return blk


def _moved(blk, x, row=0, col=0, tol=1e-6):
    """Which (b, h, w) positions move when one input token is perturbed.

    The perturbation has to survive the block's LayerNorm, so it is not a
    constant offset (LayerNorm would subtract that straight back out)."""
    with torch.no_grad():
        base = blk(x)
        x2 = x.clone()
        x2[0, row, col, :] = x[0, row, col, :] * 2.0 + torch.linspace(-3.0, 3.0, C)
        return (blk(x2) - base).abs().amax(-1) > tol


# --------------------------------------------------------------------------
# window_partition / window_reverse
# --------------------------------------------------------------------------
def check_window_partition_shape_and_layout(ns):
    x = _x()
    win = ns.window_partition(x, WS)
    assert tuple(win.shape) == (B * NW, NTOK, C), (
        f"an {H}x{W} map with {WS}x{WS} windows gives {NW} windows per image, so "
        f"the output should be (B*num_windows, ws*ws, C) = {(B * NW, NTOK, C)}; "
        f"got {tuple(win.shape)}")
    assert torch.equal(win[0], x[0, :WS, :WS].reshape(NTOK, C)), (
        "window 0 must be the top-left WSxWS block of image 0, flattened "
        "row-major inside the window")
    assert torch.equal(win[1], x[0, :WS, WS:2 * WS].reshape(NTOK, C)), (
        "windows are emitted in row-major order, so window 1 is the block to the "
        "RIGHT of window 0, not the one below it")
    assert torch.equal(win[NW], x[1, :WS, :WS].reshape(NTOK, C)), (
        f"images stack outermost: window {NW} should be image 1's top-left block")


def check_window_reverse_is_the_exact_inverse(ns):
    x = _x(1)
    back = ns.window_reverse(ns.window_partition(x, WS), WS, H, W)
    assert tuple(back.shape) == (B, H, W, C), \
        f"window_reverse should rebuild (B, H, W, C) = {(B, H, W, C)}, got {tuple(back.shape)}"
    assert torch.equal(back, x), (
        "window_reverse(window_partition(x)) must return x bit for bit — it is the "
        f"exact inverse permutation; max deviation {float((back - x).abs().max()):.6f}")


def check_window_reverse_then_partition_round_trips(ns):
    g = torch.Generator().manual_seed(2)
    win = torch.randn(B * NW, NTOK, C, generator=g)
    back = ns.window_partition(ns.window_reverse(win, WS, H, W), WS)
    assert torch.equal(back, win), (
        "going the other way round — reverse then partition — must also be exact; "
        "the two functions use mismatched permutations")


# --------------------------------------------------------------------------
# window_attention
# --------------------------------------------------------------------------
def check_window_attention_matches_scaled_dot_product(ns):
    q, k, v = _qkv(3)
    out = ns.window_attention(q, k, v)
    ref = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(C), dim=-1) @ v
    assert tuple(out.shape) == (B * NW, NTOK, C), \
        f"attention must keep the window shape {(B * NW, NTOK, C)}, got {tuple(out.shape)}"
    assert torch.allclose(out, ref, atol=1e-5), (
        "unmasked window attention is plain scaled dot-product attention inside "
        f"each window, softmax(q k^T / sqrt(C)) v; max deviation "
        f"{float((out - ref).abs().max()):.6f} (a missing 1/sqrt(C) scale is the "
        "usual cause)")


def check_attention_weights_are_a_distribution(ns):
    q, k, _ = _qkv(4)
    weights = ns.window_attention(q, k, _onehot_values())[:, :, :NTOK]
    assert bool((weights >= -1e-6).all()), \
        "attention weights come out of a softmax and cannot be negative"
    sums = weights.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"each query's attention weights must sum to 1 (softmax over the last "
        f"dim); got sums in [{float(sums.min()):.4f}, {float(sums.max()):.4f}]")


def check_attention_honours_the_shifted_window_mask(ns):
    """Ties the mask and the attention together: -inf entries must get weight 0."""
    q, k, _ = _qkv(5)
    mask = ns.create_shifted_window_mask(H, W, WS, SHIFT)
    weights = ns.window_attention(q, k, _onehot_values(), mask)[:, :, :NTOK]
    blocked = torch.isinf(mask) | (mask < -1e5)
    for i in range(B * NW):
        wm = blocked[i % NW]
        got = weights[i]
        assert float(got[wm].abs().max() if wm.any() else torch.zeros(1)) < 1e-6, (
            f"window {i}: tokens the mask marks -inf still received attention "
            f"(max weight {float(got[wm].abs().max()):.5f}). The mask is additive "
            "and must be applied to the scores BEFORE the softmax.")
        assert float(got[~wm].sum()) > 1 - 1e-4, (
            f"window {i}: the allowed tokens should carry the full probability "
            f"mass, they carry {float(got[~wm].sum()):.4f}")


def check_bias_is_added_before_the_softmax(ns):
    q, k, v = _qkv(6)
    g = torch.Generator().manual_seed(7)
    bias = torch.randn(NTOK, NTOK, generator=g)
    out = ns.window_attention(q, k, v, None, bias)
    ref = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(C) + bias, dim=-1) @ v
    assert torch.allclose(out, ref, atol=1e-5), (
        "the relative position bias is added to the scores before the softmax and "
        "is shared by every window; max deviation "
        f"{float((out - ref).abs().max()):.6f}")


def check_constant_bias_leaves_attention_unchanged(ns):
    """Softmax is shift invariant — a check no reference is needed for."""
    q, k, v = _qkv(8)
    plain = ns.window_attention(q, k, v)
    shifted = ns.window_attention(q, k, v, None, torch.full((NTOK, NTOK), 4.0))
    assert torch.allclose(plain, shifted, atol=1e-5), (
        "adding the SAME constant to every attention score must not change the "
        "softmax, so a constant bias table is a no-op; max deviation "
        f"{float((plain - shifted).abs().max()):.6f} (this usually means the bias "
        "is being added after the softmax, or to the values)")


# --------------------------------------------------------------------------
# create_shifted_window_mask
# --------------------------------------------------------------------------
def _allowed(mask):
    return ~(torch.isinf(mask) | (mask < -1e5))


def check_zero_shift_masks_nothing(ns):
    mask = ns.create_shifted_window_mask(H, W, WS, 0)
    assert tuple(mask.shape) == (NW, NTOK, NTOK), (
        f"the mask is one (tokens x tokens) block per window, "
        f"{(NW, NTOK, NTOK)}; got {tuple(mask.shape)}")
    assert bool((mask == 0).all()), (
        "with shift_size=0 nothing wraps, so every token in a window may attend to "
        "every other token — the mask must be all zeros")


def check_mask_entries_are_zero_or_minus_inf(ns):
    mask = ns.create_shifted_window_mask(H, W, WS, SHIFT)
    allowed = _allowed(mask)
    assert bool((mask[allowed] == 0).all()), \
        "allowed pairs must contribute 0 to the scores, not some other constant"
    assert bool((mask[~allowed] < -1e5).all()), \
        "blocked pairs must be -inf (or an equally large negative) so softmax zeros them"
    diag = torch.arange(NTOK)
    assert bool(allowed[:, diag, diag].all()), \
        "a token always belongs to its own region, so the diagonal is never masked"
    assert bool((allowed.sum(-1) >= 1).all()), \
        "some query can attend to nothing — softmax over an all -inf row gives NaN"
    assert torch.equal(allowed, allowed.transpose(-2, -1)), \
        "'same region' is symmetric, so the mask must equal its own transpose"


def check_mask_partitions_each_window_into_regions(ns):
    """'Allowed' has to be an equivalence relation: the regions are a partition."""
    allowed = _allowed(ns.create_shifted_window_mask(H, W, WS, SHIFT)).float()
    reach = (allowed @ allowed) > 0
    assert torch.equal(reach, allowed.bool()), (
        "if i can see j and j can see k then i and k are in the same wrapped "
        "region and must see each other — the allowed pairs must form a partition "
        "of each window's tokens, not an arbitrary pattern")


def check_shifted_mask_region_sizes(ns):
    """Geometry of an 8x8 map, 4x4 windows, shift 2: one window is untouched,
    two straddle a single seam, one straddles both."""
    allowed = _allowed(ns.create_shifted_window_mask(H, W, WS, SHIFT))
    counts = allowed.sum(-1)
    for w in range(NW):
        row = counts[w]
        assert bool((row == row[0]).all()), (
            f"window {w}: the wrapped regions tile the window evenly, so every "
            f"query should see the same number of tokens; got {row.tolist()}")
    got = sorted(int(counts[w][0]) for w in range(NW))
    assert got == [4, 8, 8, 16], (
        f"after a shift of {SHIFT} on an {H}x{W} map, the four windows should "
        f"allow 16 / 8 / 8 / 4 tokens per query (interior window, two windows "
        f"split by one seam, one split by both); got {got}")


# --------------------------------------------------------------------------
# SwinBlock
# --------------------------------------------------------------------------
def check_block_preserves_shape_and_gradients(ns):
    for shift in (0, SHIFT):
        blk = _block(ns, "SwinBlock", shift)
        x = _x(9).requires_grad_(True)
        out = blk(x)
        assert tuple(out.shape) == (B, H, W, C), (
            f"shift_size={shift}: the block maps (B, H, W, C) to itself, "
            f"{(B, H, W, C)}; got {tuple(out.shape)}")
        assert torch.isfinite(out).all(), (
            f"shift_size={shift}: output contains NaN/Inf — a fully masked "
            "attention row would do that")
        torch.manual_seed(0)
        (out * torch.randn_like(out)).sum().backward()
        assert x.grad is not None and bool((x.grad != 0).any()), \
            f"shift_size={shift}: no gradient reached the input"
        grads = [p.grad for p in blk.parameters() if p.grad is not None]
        assert grads and any(bool((g != 0).any()) for g in grads), \
            f"shift_size={shift}: no block parameter received a gradient"


def check_unshifted_block_is_window_local(ns):
    """Swap two whole windows in, get the two output windows swapped back."""
    blk = _block(ns, "SwinBlock", 0)
    x = _x(10)
    x2 = x.clone()
    x2[:, 0:WS, 0:WS] = x[:, WS:2 * WS, WS:2 * WS]
    x2[:, WS:2 * WS, WS:2 * WS] = x[:, 0:WS, 0:WS]
    with torch.no_grad():
        out, out2 = blk(x), blk(x2)
    assert torch.allclose(out2[:, 0:WS, 0:WS], out[:, WS:2 * WS, WS:2 * WS], atol=1e-5), (
        "W-MSA processes each window independently and identically, so swapping "
        "two windows of the input must swap the corresponding output windows; it "
        "did not — attention is probably reaching across window boundaries")
    assert torch.allclose(out2[:, WS:2 * WS, WS:2 * WS], out[:, 0:WS, 0:WS], atol=1e-5), \
        "the second swapped window did not come back in the other slot"
    assert torch.allclose(out2[:, 0:WS, WS:2 * WS], out[:, 0:WS, WS:2 * WS], atol=1e-5), (
        "an untouched window changed when two other windows were swapped — windows "
        "must not interact inside a single W-MSA block")


def check_unshifted_receptive_field_is_one_window(ns):
    blk = _block(ns, "SwinBlock", 0)
    moved = _moved(blk, _x(11))
    expected = torch.zeros(B, H, W, dtype=torch.bool)
    expected[0, 0:WS, 0:WS] = True
    assert torch.equal(moved, expected), (
        f"perturbing the token at (0, 0) of image 0 must move exactly its own "
        f"{WS}x{WS} window and nothing else. Moved {int(moved.sum())} positions "
        f"(expected {int(expected.sum())}); image 1 moved: "
        f"{bool(moved[1].any())}")


def check_shifted_receptive_field_follows_the_cyclic_shift(ns):
    """The corner token wraps into the last window but keeps its own region,
    so only the 2x2 corner may move."""
    blk = _block(ns, "SwinBlock", SHIFT)
    moved = _moved(blk, _x(12))
    expected = torch.zeros(B, H, W, dtype=torch.bool)
    expected[0, 0:SHIFT, 0:SHIFT] = True
    assert torch.equal(moved, expected), (
        f"after a cyclic shift of {SHIFT}, the token at (0, 0) lands in the "
        f"wrapped corner window, where the mask only lets it mix with the other "
        f"{SHIFT}x{SHIFT} corner tokens. Exactly those {SHIFT * SHIFT} positions "
        f"should move; {int(moved.sum())} did. Too many means the mask is not "
        "being applied, a different set means the roll direction or the undo-roll "
        "is off.")


def check_shifting_changes_the_computation(ns):
    """Same weights, same input, different shift -> different answer."""
    a = _block(ns, "SwinBlock", 0, seed=42)
    b = _block(ns, "SwinBlock", SHIFT, seed=42)
    x = _x(13)
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x), atol=1e-5), (
            "a block with shift_size=0 and one with shift_size=%d produced the "
            "same output from identical weights — the cyclic shift is not being "
            "applied" % SHIFT)


# --------------------------------------------------------------------------
# relative_position_index
# --------------------------------------------------------------------------
def check_relative_index_shape_and_range(ns):
    rpi = ns.relative_position_index(WS)
    assert tuple(rpi.shape) == (NTOK, NTOK), \
        f"one table index per (query, key) pair: {(NTOK, NTOK)}, got {tuple(rpi.shape)}"
    assert rpi.dtype in (torch.long, torch.int64), \
        f"the index map is used to gather a table, so it must be integer; got {rpi.dtype}"
    assert int(rpi.min()) >= 0 and int(rpi.max()) < TABLE, (
        f"offsets run from -(ws-1) to +(ws-1) in each axis, so indices must fall in "
        f"[0, (2*{WS}-1)**2 = {TABLE}); got [{int(rpi.min())}, {int(rpi.max())}]")


def check_relative_index_depends_only_on_the_offset(ns):
    rpi = ns.relative_position_index(WS)
    coords = torch.stack(torch.meshgrid(torch.arange(WS), torch.arange(WS),
                                        indexing="ij"), -1).reshape(NTOK, 2)
    offsets = coords[:, None, :] - coords[None, :, :]
    seen = {}
    for i in range(NTOK):
        for j in range(NTOK):
            key = (int(offsets[i, j, 0]), int(offsets[i, j, 1]))
            idx = int(rpi[i, j])
            if key in seen:
                assert seen[key] == idx, (
                    f"pairs with the same relative offset {key} got different table "
                    f"entries ({seen[key]} and {idx}) — the bias must depend on the "
                    "offset alone, that is what makes it translation invariant")
            seen[key] = idx
    assert len(set(seen.values())) == len(seen) == TABLE, (
        f"there are {TABLE} distinct relative offsets in a {WS}x{WS} window and "
        f"each needs its own table row; the map produced {len(set(seen.values()))} "
        f"distinct indices for {len(seen)} offsets")


def check_relative_index_diagonal_is_the_zero_offset(ns):
    rpi = ns.relative_position_index(WS)
    diag = rpi.diagonal()
    assert bool((diag == diag[0]).all()), (
        f"every token's offset to itself is (0, 0), so the whole diagonal must be "
        f"one constant index; got {diag.tolist()}")
    assert int(rpi[0, 1]) != int(rpi[1, 0]), \
        "opposite offsets are different offsets and need different table entries"


# --------------------------------------------------------------------------
# SwinBlockRelPosBias
# --------------------------------------------------------------------------
def _bias_table(ns, blk):
    plain = {n for n, _ in _block(ns, "SwinBlock", 0).named_parameters()}
    extra = [(n, p) for n, p in blk.named_parameters() if n not in plain]
    assert extra, (
        "SwinBlockRelPosBias has no parameters beyond the plain SwinBlock — the "
        "relative position bias table must be a learned nn.Parameter")
    tbl = [p for _, p in extra if p.numel() == TABLE]
    assert tbl, (
        f"expected a learned bias table with (2*window_size-1)**2 = {TABLE} entries "
        f"(one per relative offset); the extra parameters are "
        f"{[(n, tuple(p.shape)) for n, p in extra]}")
    return tbl[0]


def check_biased_block_shape_and_table(ns):
    for shift in (0, SHIFT):
        blk = _block(ns, "SwinBlockRelPosBias", shift)
        tbl = _bias_table(ns, blk)
        with torch.no_grad():
            tbl.copy_(torch.randn(tbl.shape))
        x = _x(14)
        out = blk(x)
        assert tuple(out.shape) == (B, H, W, C), (
            f"shift_size={shift}: the biased block must still map (B, H, W, C) to "
            f"itself; got {tuple(out.shape)}")
        assert torch.isfinite(out).all(), \
            f"shift_size={shift}: output contains NaN/Inf"
        torch.manual_seed(0)
        (out * torch.randn_like(out)).sum().backward()
        assert tbl.grad is not None and float(tbl.grad.abs().sum()) > 0, (
            f"shift_size={shift}: the bias table received no gradient — it is not "
            "on the path from the attention scores to the output, so it can never "
            "be learned")


def check_constant_bias_table_is_a_no_op(ns):
    """Softmax shift invariance again, now through the whole block."""
    blk = _block(ns, "SwinBlockRelPosBias", SHIFT, seed=21)
    tbl = _bias_table(ns, blk)
    x = _x(15)
    with torch.no_grad():
        tbl.zero_()
        zero = blk(x)
        tbl.fill_(3.5)
        const = blk(x)
        tbl.copy_(torch.randn(tbl.shape, generator=torch.Generator().manual_seed(1)))
        rand = blk(x)
    assert torch.allclose(zero, const, atol=1e-5), (
        "filling the bias table with a single constant adds the same number to "
        "every attention score, which softmax ignores — the block's output must "
        f"not change; max deviation {float((zero - const).abs().max()):.6f}")
    assert not torch.allclose(zero, rand, atol=1e-5), (
        "randomising the bias table changed nothing — the table is not reaching "
        "the attention scores at all")


def check_biased_block_is_still_window_local(ns):
    """The bias is shared across windows, so window locality must survive."""
    blk = _block(ns, "SwinBlockRelPosBias", 0, seed=22)
    tbl = _bias_table(ns, blk)
    with torch.no_grad():
        tbl.copy_(torch.randn(tbl.shape, generator=torch.Generator().manual_seed(2)))
    moved = _moved(blk, _x(16))
    expected = torch.zeros(B, H, W, dtype=torch.bool)
    expected[0, 0:WS, 0:WS] = True
    assert torch.equal(moved, expected), (
        f"with shift_size=0 the biased block must still only mix tokens inside one "
        f"{WS}x{WS} window (the same bias applies to every window); perturbing "
        f"(0, 0) moved {int(moved.sum())} positions instead of {int(expected.sum())}")


CHECKS = [
    check_window_partition_shape_and_layout,
    check_window_reverse_is_the_exact_inverse,
    check_window_reverse_then_partition_round_trips,
    check_window_attention_matches_scaled_dot_product,
    check_attention_weights_are_a_distribution,
    check_attention_honours_the_shifted_window_mask,
    check_bias_is_added_before_the_softmax,
    check_constant_bias_leaves_attention_unchanged,
    check_zero_shift_masks_nothing,
    check_mask_entries_are_zero_or_minus_inf,
    check_mask_partitions_each_window_into_regions,
    check_shifted_mask_region_sizes,
    check_block_preserves_shape_and_gradients,
    check_unshifted_block_is_window_local,
    check_unshifted_receptive_field_is_one_window,
    check_shifted_receptive_field_follows_the_cyclic_shift,
    check_shifting_changes_the_computation,
    check_relative_index_shape_and_range,
    check_relative_index_depends_only_on_the_offset,
    check_relative_index_diagonal_is_the_zero_offset,
    check_biased_block_shape_and_table,
    check_constant_bias_table_is_a_no_op,
    check_biased_block_is_still_window_local,
]
