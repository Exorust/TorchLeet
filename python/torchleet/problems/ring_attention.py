"""ring-attention — sequence-parallel attention with online softmax across devices.

Two anchors, both independent of how the ring is written:

  * with num_devices=1 the ring degenerates to ordinary attention, and for any
    number of devices the result must still equal ordinary attention. The oracle
    is F.scaled_dot_product_attention, which ships with the user's PyTorch.
  * ring_step is checked for its own online-softmax property: feeding one K,V
    block must give plain softmax attention over that block, and feeding a block
    in two halves must give exactly the same answer as feeding it whole. That is
    the whole point of the running max/sum, and it is checked with the solver's
    own ring_step both times.
"""
import math

import torch
import torch.nn.functional as F

from torchleet.runner import Skip

ENTRIES = ["ring_step", "RingAttention"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "ring_step is FlashAttention's inner loop: new_max = max(running_max, "
    "S.max(-1)), correction = exp(running_max - new_max), then rescale BOTH the "
    "running sum and the running output by that correction before adding the block.",
    "Divide by running_sum only once, at the very end — never inside the loop.",
    "Causal masking is per chunk pair: skip a K,V chunk whose index is greater "
    "than the Q chunk's (all future), apply a triangular mask when the indices are "
    "equal, and mask nothing when the K,V chunk is in the past.",
]

SEQ, HEAD_DIM = 24, 16


def _qkv(seed=0, n=SEQ, d=HEAD_DIM):
    torch.manual_seed(seed)
    return torch.randn(n, d), torch.randn(n, d), torch.randn(n, d)


def _reference(q, k, v, causal=False):
    """torch's own fused attention, used as the oracle."""
    return F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=causal).squeeze(0)


def _accumulators(rows, d):
    return (torch.full((rows, 1), float("-inf")), torch.zeros(rows, 1), torch.zeros(rows, d))


def check_ring_step_is_one_block_of_attention(ns):
    """Starting from empty accumulators, one step normalized must be plain attention."""
    q, k, v = _qkv(0, n=6, d=HEAD_DIM)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    m, s, o = ns.ring_step(q, k, v, *_accumulators(6, HEAD_DIM), scale)
    for name, t, shape in (("running_max", m, (6, 1)), ("running_sum", s, (6, 1)),
                           ("running_output", o, (6, HEAD_DIM))):
        assert tuple(t.shape) == shape, \
            f"ring_step should return {name} with shape {shape}, got {tuple(t.shape)}"
    got = o / s
    expected = torch.softmax(q @ k.T * scale, dim=-1) @ v
    diff = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=1e-5), (
        f"after a single ring step, running_output / running_sum must be the plain "
        f"softmax attention of local_q over that K,V block (max diff {diff:.2e})")


def check_online_softmax_splits_correctly(ns):
    """Two half blocks must accumulate to exactly what one whole block gives."""
    torch.manual_seed(1)
    q = torch.randn(6, HEAD_DIM)
    k, v = torch.randn(12, HEAD_DIM), torch.randn(12, HEAD_DIM)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    m, s, o = _accumulators(6, HEAD_DIM)
    m, s, o = ns.ring_step(q, k[:6], v[:6], m, s, o, scale)
    m, s, o = ns.ring_step(q, k[6:], v[6:], m, s, o, scale)
    two_pass = o / s
    whole = torch.softmax(q @ k.T * scale, dim=-1) @ v
    diff = (two_pass - whole).abs().max().item()
    assert torch.allclose(two_pass, whole, atol=1e-5), (
        f"attending to a K,V block in two halves disagrees with attending to it in "
        f"one go (max diff {diff:.2e}) — the running sum and running output must "
        f"both be rescaled by exp(old_max - new_max) when the max moves")


def check_ring_step_is_numerically_stable(ns):
    """Large scores must not overflow — that is what the running max is for."""
    torch.manual_seed(2)
    q = torch.randn(6, HEAD_DIM) * 40
    k, v = torch.randn(12, HEAD_DIM), torch.randn(12, HEAD_DIM)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    m, s, o = _accumulators(6, HEAD_DIM)
    m, s, o = ns.ring_step(q, k[:6], v[:6], m, s, o, scale)
    m, s, o = ns.ring_step(q, k[6:], v[6:], m, s, o, scale)
    got = o / s
    assert torch.isfinite(got).all(), \
        "NaN/Inf on large scores — subtract the running max before exp()"
    expected = torch.softmax(q @ k.T * scale, dim=-1) @ v
    diff = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=1e-4), \
        f"wrong values on large scores (max diff {diff:.2e})"


def _forward(attn, q, k, v, causal):
    fn = getattr(attn, "forward", attn)
    return fn(q, k, v, causal=causal)


def check_single_device_equals_full_attention(ns):
    """One device means one chunk: the ring must collapse to ordinary attention."""
    q, k, v = _qkv(3)
    out = _forward(ns.RingAttention(num_devices=1), q, k, v, False)
    assert tuple(out.shape) == (SEQ, HEAD_DIM), \
        f"output should be {(SEQ, HEAD_DIM)}, got {tuple(out.shape)}"
    expected = _reference(q, k, v)
    diff = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, atol=1e-5), (
        f"with num_devices=1 there is nothing to distribute, so the result must "
        f"equal ordinary attention (max diff {diff:.2e})")


def check_sharding_does_not_change_the_answer(ns):
    """Splitting the sequence over more devices must not move the output."""
    q, k, v = _qkv(4)
    expected = _reference(q, k, v)
    for n_dev in (2, 3, 4, 6):
        out = _forward(ns.RingAttention(num_devices=n_dev), q, k, v, False)
        diff = (out - expected).abs().max().item()
        assert torch.allclose(out, expected, atol=1e-5), (
            f"{n_dev} devices gives a different answer from full attention (max diff "
            f"{diff:.2e}) — after {n_dev} rotations every Q chunk must have seen "
            f"every K,V chunk exactly once")


def check_causal_matches_masked_attention(ns):
    q, k, v = _qkv(5)
    expected = _reference(q, k, v, causal=True)
    for n_dev in (1, 2, 4, 6):
        out = _forward(ns.RingAttention(num_devices=n_dev), q, k, v, True)
        diff = (out - expected).abs().max().item()
        assert torch.allclose(out, expected, atol=1e-5), (
            f"causal ring attention with {n_dev} devices disagrees with causally "
            f"masked full attention (max diff {diff:.2e}) — a K,V chunk ahead of the "
            f"Q chunk must be skipped entirely, and the diagonal chunk needs a "
            f"triangular mask")


def check_causal_hides_the_future(ns):
    """Guards against `causal=True` being accepted and then ignored."""
    q, k, v = _qkv(6)
    attn = ns.RingAttention(num_devices=4)
    causal = _forward(attn, q, k, v, True)
    v2 = v.clone()
    v2[SEQ // 2:] = torch.randn(SEQ - SEQ // 2, HEAD_DIM)
    causal2 = _forward(ns.RingAttention(num_devices=4), q, k, v2, True)
    half = SEQ // 2
    diff = (causal[:half] - causal2[:half]).abs().max().item()
    assert torch.allclose(causal[:half], causal2[:half], atol=1e-5), (
        f"rewriting the second half of V changed the outputs of the first half "
        f"(max diff {diff:.2e}) — with causal=True those queries must not attend "
        f"to later positions")


def check_ring_rotation_returns_to_start(ns):
    """n rotations of an n-device ring is the identity; one rotation is not."""
    attn = ns.RingAttention(num_devices=4)
    rotate = getattr(attn, "_rotate_ring", None)
    if rotate is None:
        raise Skip("no _rotate_ring on this RingAttention")
    start = [f"chunk_{i}" for i in range(4)]
    once = list(rotate(list(start)))
    assert once != start, \
        f"_rotate_ring left the chunks where they were: {once}"
    chunks = list(start)
    for _ in range(4):
        chunks = list(rotate(chunks))
    assert chunks == start, (
        f"after 4 rotations of a 4-device ring every device should hold its own "
        f"chunk again, got {chunks}")


CHECKS = [
    check_ring_step_is_one_block_of_attention,
    check_online_softmax_splits_correctly,
    check_ring_step_is_numerically_stable,
    check_single_device_equals_full_attention,
    check_sharding_does_not_change_the_answer,
    check_causal_matches_masked_attention,
    check_causal_hides_the_future,
    check_ring_rotation_returns_to_start,
]
