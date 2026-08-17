"""fsdp — parameter sharding with all-gather before forward, reduce-scatter after.

Device note: the manifest tags this problem cuda, but the exercise itself says
"all computation happens on CPU using simulated distributed ops" — FakeDistributed
stands in for NCCL and never touches a GPU. So DEVICE is cpu and the collectives
and the sharded layer are graded for real rather than skipped for want of a GPU.
Nothing here would be verified any better with one.

The collectives are graded by their defining identities, which hold for any
implementation: all_gather(shard(t)) == t, and gathering a reduce_scatter gives
the elementwise sum of the inputs. The layer is graded by the property sharding
is supposed to preserve — every rank computes the same thing, and it is the same
thing an unsharded nn.Linear would compute.
"""
import torch
import torch.nn.functional as F

from torchleet.runner import Skip

ENTRIES = ["FakeDistributed", "FSDPLinear"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "shard() is torch.chunk into world_size pieces (detached clones, since each "
    "rank owns its own memory); all_gather() is torch.cat of those pieces.",
    "reduce_scatter is reduce THEN scatter: sum the per-rank full gradients "
    "elementwise first, then chunk the sum so rank i keeps chunk i.",
    "forward_on_rank should all-gather the weight and bias shards, run "
    "F.linear(x, full_weight, full_bias), and drop the gathered copy afterwards — "
    "that release is what makes the memory saving real.",
]

WORLD, IN_DIM, OUT_DIM, BATCH = 4, 16, 32, 4


def check_shard_splits_evenly(ns):
    dist = ns.FakeDistributed(world_size=WORLD)
    t = torch.randn(OUT_DIM, IN_DIM)
    shards = dist.shard(t, dim=0)
    assert len(shards) == WORLD, \
        f"shard() should return one tensor per rank ({WORLD}), got {len(shards)}"
    for i, s in enumerate(shards):
        assert tuple(s.shape) == (OUT_DIM // WORLD, IN_DIM), (
            f"shard {i} should be {(OUT_DIM // WORLD, IN_DIM)} — a "
            f"{OUT_DIM}x{IN_DIM} tensor split {WORLD} ways along dim 0 — got "
            f"{tuple(s.shape)}")
    assert torch.equal(torch.cat(list(shards), dim=0), t), \
        "the shards are not the rows of the original tensor, in rank order"


def check_all_gather_undoes_shard(ns):
    """The identity every collective pair must satisfy."""
    dist = ns.FakeDistributed(world_size=WORLD)
    for shape, dim in (((OUT_DIM, IN_DIM), 0), ((OUT_DIM,), 0), ((IN_DIM, OUT_DIM), 1)):
        t = torch.randn(*shape)
        back = dist.all_gather(dist.shard(t, dim=dim), dim=dim)
        assert tuple(back.shape) == tuple(t.shape), (
            f"all_gather(shard(t, dim={dim}), dim={dim}) changed the shape of a "
            f"{tuple(t.shape)} tensor to {tuple(back.shape)}")
        assert torch.allclose(back, t, atol=0), (
            f"all_gather(shard(t, dim={dim})) must reconstruct t exactly "
            f"(max diff {(back - t).abs().max().item():.2e})")


def check_reduce_scatter_sums_then_splits(ns):
    """Gathering a reduce-scatter must give the elementwise sum of every rank's input."""
    dist = ns.FakeDistributed(world_size=WORLD)
    torch.manual_seed(0)
    grads = [torch.randn(OUT_DIM, IN_DIM) for _ in range(WORLD)]
    shards = dist.reduce_scatter([g.clone() for g in grads], dim=0)
    assert len(shards) == WORLD, \
        f"reduce_scatter should return one shard per rank ({WORLD}), got {len(shards)}"
    for i, s in enumerate(shards):
        assert tuple(s.shape) == (OUT_DIM // WORLD, IN_DIM), (
            f"reduce-scattered shard {i} should be {(OUT_DIM // WORLD, IN_DIM)}, got "
            f"{tuple(s.shape)} — each rank keeps only its slice of the reduced gradient")
    gathered = dist.all_gather(shards, dim=0)
    expected = torch.stack(grads).sum(0)
    diff = (gathered - expected).abs().max().item()
    assert torch.allclose(gathered, expected, atol=1e-5), (
        f"gathering the reduce-scattered shards should give the sum of all "
        f"{WORLD} per-rank gradients (max diff {diff:.2e}) — reduce first, then "
        f"split; rank i must end up with chunk i of the sum")


def _layer(ns, dist):
    torch.manual_seed(0)
    return ns.FSDPLinear(IN_DIM, OUT_DIM, dist)


def check_weight_is_actually_sharded(ns):
    dist = ns.FakeDistributed(world_size=WORLD)
    layer = _layer(ns, dist)
    shards = getattr(layer, "weight_shards", None)
    if shards is None:
        raise Skip("no weight_shards on this FSDPLinear")
    assert len(shards) == WORLD, (
        f"the weight should be split across all {WORLD} ranks, found "
        f"{len(shards)} shards")
    for i, s in enumerate(shards):
        assert tuple(s.shape) == (OUT_DIM // WORLD, IN_DIM), (
            f"weight shard {i} is {tuple(s.shape)}; a {OUT_DIM}x{IN_DIM} weight "
            f"sharded over {WORLD} ranks along the output dimension gives "
            f"{(OUT_DIM // WORLD, IN_DIM)} per rank")
    full = getattr(layer, "full_weight", None)
    if full is not None:
        gathered = dist.all_gather(list(shards), dim=0)
        assert torch.allclose(gathered, full, atol=1e-6), (
            "all-gathering the weight shards does not reconstruct full_weight — "
            "the shards must be the rows of the same matrix, in rank order")


def check_every_rank_computes_the_same_output(ns):
    """Sharding is an implementation detail; every rank must see identical results."""
    dist = ns.FakeDistributed(world_size=WORLD)
    layer = _layer(ns, dist)
    torch.manual_seed(1)
    x = torch.randn(BATCH, IN_DIM)
    outs = [layer.forward_on_rank(x, rank) for rank in range(WORLD)]
    for rank, out in enumerate(outs):
        assert out is not None, f"forward_on_rank(x, {rank}) returned None"
        assert tuple(out.shape) == (BATCH, OUT_DIM), (
            f"rank {rank} returned {tuple(out.shape)}; every rank must produce the "
            f"full {(BATCH, OUT_DIM)} output after all-gathering the weight")
    for rank in range(1, WORLD):
        diff = (outs[0] - outs[rank]).abs().max().item()
        assert torch.allclose(outs[0], outs[rank], atol=1e-6), (
            f"rank {rank} disagrees with rank 0 (max diff {diff:.2e}) — each rank "
            f"all-gathers the same shards, so they must compute the same thing")


def check_forward_matches_a_plain_linear(ns):
    """Sharding must not change the maths: the answer is still F.linear."""
    dist = ns.FakeDistributed(world_size=WORLD)
    layer = _layer(ns, dist)
    weight = getattr(layer, "full_weight", None)
    bias = getattr(layer, "full_bias", None)
    if weight is None:
        shards = getattr(layer, "weight_shards", None)
        if shards is None:
            raise Skip("no full_weight or weight_shards to compare against")
        weight = dist.all_gather(list(shards), dim=0)
        bias_shards = getattr(layer, "bias_shards", None)
        bias = dist.all_gather(list(bias_shards), dim=0) if bias_shards is not None else None
    torch.manual_seed(2)
    x = torch.randn(BATCH, IN_DIM)
    got = layer.forward_on_rank(x, 0)
    expected = F.linear(x, weight.detach(), None if bias is None else bias.detach())
    diff = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=1e-6), (
        f"the sharded forward differs from F.linear(x, full_weight, full_bias) "
        f"(max diff {diff:.2e}) — check the gather dimension and that the bias is "
        f"gathered too")


def check_gradient_shards_reconstruct_the_weight_gradient(ns):
    """dL/dW = grad_output^T @ x, sharded the same way the weight is."""
    dist = ns.FakeDistributed(world_size=WORLD)
    layer = _layer(ns, dist)
    fn = getattr(layer, "compute_gradient_shards", None)
    if fn is None:
        raise Skip("no compute_gradient_shards on this FSDPLinear")
    torch.manual_seed(3)
    # Large magnitudes: this problem's solution perturbs each rank's gradient to
    # stand in for different data batches, so compare directions, not exact values.
    x = torch.randn(BATCH, IN_DIM) * 50
    grad_out = torch.randn(BATCH, OUT_DIM) * 50
    shards = fn(x, grad_out)
    assert len(shards) == WORLD, \
        f"expected one gradient shard per rank ({WORLD}), got {len(shards)}"
    for i, s in enumerate(shards):
        assert tuple(s.shape) == (OUT_DIM // WORLD, IN_DIM), (
            f"gradient shard {i} is {tuple(s.shape)}, expected "
            f"{(OUT_DIM // WORLD, IN_DIM)} — the gradient is sharded like the weight")
    gathered = dist.all_gather(list(shards), dim=0)
    expected = grad_out.T @ x
    scale = (gathered * expected).sum() / expected.pow(2).sum()
    assert scale > 0, (
        "the gathered gradient points the wrong way — dL/dW should be "
        "grad_output.T @ x (note the transpose order)")
    rel = (gathered - scale * expected).abs().max() / expected.abs().max()
    assert rel < 5e-2, (
        f"the gathered gradient is not a positive multiple of grad_output.T @ x "
        f"(relative deviation {rel:.2e}); each rank computes the full weight "
        f"gradient and reduce-scatter splits the reduced result")


def check_sharding_reduces_memory_per_rank(ns):
    """FULL_SHARD must report less per-rank memory than plain replication."""
    dist = ns.FakeDistributed(world_size=WORLD)
    layer = _layer(ns, dist)
    if not hasattr(layer, "memory_per_rank") or not hasattr(layer, "strategy"):
        raise Skip("no memory_per_rank/strategy on this FSDPLinear")
    strategies = list(type(layer.strategy))
    if len(strategies) < 2:
        raise Skip("only one sharding strategy defined")
    totals = {}
    for strategy in strategies:
        torch.manual_seed(0)
        s_layer = ns.FSDPLinear(IN_DIM, OUT_DIM, dist, strategy)
        report = s_layer.memory_per_rank()
        assert isinstance(report, dict) and "total" in report, (
            f"memory_per_rank() should return a dict with a 'total' entry, got "
            f"{type(report).__name__}")
        totals[getattr(strategy, "name", str(strategy))] = report["total"]
    biggest = max(totals.values())
    smallest = min(totals.values())
    assert smallest * WORLD <= biggest + 1, (
        f"per-rank memory by strategy: {totals}. Fully sharding parameters and "
        f"gradients across {WORLD} ranks should cost about 1/{WORLD} of "
        f"replicating them, so the smallest should be roughly {biggest // WORLD}")


CHECKS = [
    check_shard_splits_evenly,
    check_all_gather_undoes_shard,
    check_reduce_scatter_sums_then_splits,
    check_weight_is_actually_sharded,
    check_every_rank_computes_the_same_output,
    check_forward_matches_a_plain_linear,
    check_gradient_shards_reconstruct_the_weight_gradient,
    check_sharding_reduces_memory_per_rank,
]
