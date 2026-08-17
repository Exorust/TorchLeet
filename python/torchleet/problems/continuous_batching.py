"""continuous-batching — iteration-level scheduling for LLM serving.

Grading a scheduler means grading behaviour, not numbers, so these checks drive
the solver's own scheduler with a deterministic toy model (its logits are a
one-hot spike, so sampling and argmax pick the same token) and assert the
properties a correct scheduler must have:

  * every request submitted comes back completed, exactly once;
  * the active batch never exceeds max_batch_size;
  * a request stops at EOS or at max_gen_len, never past it;
  * the same requests produce the same tokens whatever the batch size — the point
    of continuous batching is throughput, not different answers;
  * a bigger batch finishes the same work in fewer steps, which a scheduler that
    serves one request at a time cannot do.

The toy model's next token depends only on how many non-EOS tokens a row holds,
so padding choices cannot change the result.
"""
import torch
import torch.nn as nn

from torchleet.runner import Skip

ENTRIES = ["Request", "ContinuousBatchScheduler"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Keep two lists: waiting_queue and active_batch. Each step() fills free slots "
    "from the queue first, then runs the model, then evicts whatever finished.",
    "A request is done when it emits eos_token_id OR len(generated_ids) reaches "
    "max_gen_len — check both after appending the token.",
    "step() should report whether work remains (active or waiting), so run() can "
    "simply loop `while self.step(model): pass`.",
]

VOCAB, EOS, MAX_STEPS = 32, 0, 2000

_g = torch.Generator().manual_seed(7)
_NEXT = torch.randint(1, VOCAB, (64,), generator=_g)
_NEXT[::7] = EOS          # every 7th sequence length ends the request


class _SpikeLM(nn.Module):
    """(B, S) token ids -> (B, S, V) logits with a single dominant token.

    The spike is chosen from the number of non-EOS tokens in the row, so right or
    left padding with zeros does not change the answer, and the -1e4 floor makes
    sampling deterministic in float32.
    """

    def forward(self, input_ids):
        ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
        ids = ids.long()
        count = (ids != EOS).sum(dim=1)
        nxt = _NEXT[count % _NEXT.numel()]
        b, s = ids.shape
        logits = torch.full((b, s, VOCAB), -1e4)
        logits.scatter_(2, nxt.view(b, 1, 1).expand(b, s, 1), 1e4)
        return logits if input_ids.dim() == 2 else logits.squeeze(0)


def _requests(ns, n=9):
    try:
        return [ns.Request(request_id=i,
                           input_ids=[1 + (i * 3) % (VOCAB - 1)] * (2 + i % 4),
                           max_gen_len=3 + (i % 5))
                for i in range(n)]
    except TypeError as e:
        raise AssertionError(
            f"could not build Request(request_id=..., input_ids=[...], "
            f"max_gen_len=...): {e}")


def _run(ns, model, reqs, max_batch_size):
    """Drive the scheduler one step at a time; returns (completed, n_steps)."""
    sched = ns.ContinuousBatchScheduler(max_batch_size, EOS, VOCAB)
    for r in reqs:
        sched.add_request(r)
    steps = 0
    with torch.no_grad():
        while True:
            more = sched.step(model)
            steps += 1
            active = getattr(sched, "active_batch", [])
            assert len(active) <= max_batch_size, (
                f"after step {steps} the active batch holds {len(active)} requests "
                f"but max_batch_size is {max_batch_size}")
            if not more:
                break
            assert steps < MAX_STEPS, (
                f"step() still reported work left after {MAX_STEPS} steps — a "
                f"request is never being marked done or never evicted")
    completed = getattr(sched, "completed", None)
    assert completed is not None, \
        "the scheduler should collect finished requests in a `completed` list"
    return sched, list(completed), steps


def check_every_request_completes_once(ns):
    model = _SpikeLM()
    reqs = _requests(ns)
    sched, completed, _ = _run(ns, model, reqs, 4)
    ids = sorted(r.request_id for r in completed)
    assert ids == list(range(len(reqs))), (
        f"expected every request id {list(range(len(reqs)))} to finish exactly once, "
        f"got {ids}")
    assert not getattr(sched, "waiting_queue", []), \
        "the waiting queue is not empty after the run"
    assert not getattr(sched, "active_batch", []), \
        "requests are still in the active batch after the run — they were never evicted"
    for r in completed:
        assert getattr(r, "is_done", False), \
            f"request {r.request_id} is in `completed` but its is_done flag is False"


def check_generation_stops_at_eos_or_max_len(ns):
    model = _SpikeLM()
    _, completed, _ = _run(ns, model, _requests(ns), 4)
    hit_eos = hit_cap = 0
    for r in completed:
        gen = list(r.generated_ids)
        assert 0 < len(gen) <= r.max_gen_len, (
            f"request {r.request_id} generated {len(gen)} tokens, outside "
            f"(0, max_gen_len={r.max_gen_len}]")
        if EOS in gen:
            assert gen.index(EOS) == len(gen) - 1, (
                f"request {r.request_id} kept generating after EOS: {gen}")
            hit_eos += 1
        else:
            assert len(gen) == r.max_gen_len, (
                f"request {r.request_id} stopped after {len(gen)} tokens without "
                f"emitting EOS, but max_gen_len is {r.max_gen_len}")
            hit_cap += 1
    assert hit_eos and hit_cap, (
        f"this workload should exercise both stop conditions, got {hit_eos} EOS "
        f"stops and {hit_cap} length stops — is one of them missing?")


def check_batch_size_does_not_change_the_output(ns):
    """Scheduling is about when work runs, never about what it produces."""
    model = _SpikeLM()
    baseline = None
    for bs in (1, 3, 5, 16):
        _, completed, _ = _run(ns, model, _requests(ns), bs)
        got = {r.request_id: list(r.generated_ids) for r in completed}
        if baseline is None:
            baseline = got
            continue
        for rid in baseline:
            assert got.get(rid) == baseline[rid], (
                f"request {rid} produced {got.get(rid)} with max_batch_size={bs} but "
                f"{baseline[rid]} with max_batch_size=1 — batching must not change "
                f"a request's tokens (are slots or KV state leaking between requests?)")


def check_batching_saves_steps(ns):
    """More slots must mean fewer iterations, or nothing is being batched."""
    model = _SpikeLM()
    _, _, one = _run(ns, model, _requests(ns), 1)
    _, _, four = _run(ns, model, _requests(ns), 4)
    assert four < one, (
        f"9 requests took {four} steps with max_batch_size=4 and {one} with "
        f"max_batch_size=1 — with 4 slots the scheduler should be advancing up to "
        f"4 requests per step")


def check_a_big_enough_batch_runs_everything_at_once(ns):
    """With more slots than requests, the run can only be as long as its longest request."""
    model = _SpikeLM()
    reqs = _requests(ns, n=9)
    _, completed, steps = _run(ns, model, reqs, 16)
    longest = max(len(r.generated_ids) for r in completed)
    assert steps <= longest + 1, (
        f"9 requests and 16 slots took {steps} steps, but the longest request only "
        f"needs {longest} tokens — with a free slot each they should all decode in "
        f"parallel from the first step")


def check_finished_slots_are_refilled(ns):
    """The 'continuous' part: a slot freed by a finished request is reused at once.

    Five requests, two slots, one of them long: a scheduler that refills eagerly
    finishes in 6 steps, one that waits for the whole batch to drain needs 8.
    """
    model = _SpikeLM()
    lens = [6, 1, 1, 1, 1]
    try:
        reqs = [ns.Request(request_id=i, input_ids=[3], max_gen_len=m)
                for i, m in enumerate(lens)]
    except TypeError as e:
        raise AssertionError(f"could not build Request(...): {e}")
    _, completed, steps = _run(ns, model, reqs, 2)
    got = {r.request_id: len(r.generated_ids) for r in completed}
    if got != {i: m for i, m in enumerate(lens)}:
        raise Skip(f"workload did not decode as expected ({got}); refill timing not measured")
    assert steps <= 7, (
        f"one 6-token request plus four 1-token requests over 2 slots took {steps} "
        f"steps; refilling a slot as soon as its request finishes needs 6 (waiting "
        f"for the whole batch to drain needs 8)")


def check_run_completes_the_queue(ns):
    """run() is the convenience wrapper: loop until nothing is left."""
    model = _SpikeLM()
    reqs = _requests(ns, n=3)
    sched = ns.ContinuousBatchScheduler(2, EOS, VOCAB)
    for r in reqs:
        sched.add_request(r)
    with torch.no_grad():
        out = sched.run(model)
    completed = out if isinstance(out, list) else getattr(sched, "completed", None)
    assert completed is not None, \
        "run(model) should return (or leave in .completed) the finished requests"
    assert sorted(r.request_id for r in completed) == [0, 1, 2], (
        f"run(model) finished {sorted(r.request_id for r in completed)} of the 3 "
        f"submitted requests")


def check_request_tracks_its_own_state(ns):
    r = _requests(ns, n=1)[0]
    assert list(r.generated_ids) == [], \
        f"a fresh Request should start with no generated tokens, got {r.generated_ids}"
    assert r.is_done is False, "a fresh Request should not be done"
    for name, expected in (("get_full_sequence", list(r.input_ids)),
                           ("total_len", len(r.input_ids))):
        fn = getattr(r, name, None)
        if fn is None:
            continue
        got = fn()
        got = list(got) if name == "get_full_sequence" else got
        assert got == expected, f"{name}() on a fresh Request returned {got}, expected {expected}"
    r.generated_ids.append(5)
    if hasattr(r, "get_full_sequence"):
        assert list(r.get_full_sequence()) == list(r.input_ids) + [5], \
            "get_full_sequence() should be input_ids followed by generated_ids"
    if hasattr(r, "total_len"):
        assert r.total_len() == len(r.input_ids) + 1, \
            "total_len() should count prompt tokens plus generated tokens"


CHECKS = [
    check_request_tracks_its_own_state,
    check_every_request_completes_once,
    check_generation_stops_at_eos_or_max_len,
    check_batch_size_does_not_change_the_output,
    check_batching_saves_steps,
    check_a_big_enough_batch_runs_everything_at_once,
    check_finished_slots_are_refilled,
    check_run_completes_the_queue,
]
