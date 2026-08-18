"""beam-search — keep the top-k partial sequences by cumulative log-probability.

The load-bearing checks use a tiny deterministic language model built as a *trap*:
the winning path loses on its first step and on its last, and only wins on the
total. Two properties follow from the definition of beam search alone, so any
correct implementation passes regardless of how it stores or prunes its beams:

  * beam_width=1 is greedy decoding (oracle: argmax of the model's own logits),
  * beam_width>=2 returns the sequence with the highest cumulative log-prob,
    where "highest" is settled by brute-force enumeration, not by a second beam
    search. Scoring candidates on the current step instead of the running total
    returns the greedy path here, half a nat short, so that bug cannot pass.
"""
import torch

ENTRIES = ["beam_search"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Keep each beam as (sequence, cumulative_log_prob) and start from "
    "([start_token], 0.0) — the start token counts towards max_len.",
    "Score a candidate with the RUNNING total: score + log_softmax(logits)[token]. "
    "Comparing only the current step's log-prob turns beam search into greedy.",
    "Pool the candidates from every beam into one list, sort by score, keep the "
    "top beam_width. At the end return the sequence of the highest-scoring beam.",
]

VOCAB = 8
MAX_LEN = 4


def _table():
    """Logits row t = the model's output for token t. Two paths, deliberately crossed.

        0 -> 1 -> 4 -> 7   log-probs -0.82, -1.60, -0.02   total -2.44
        0 -> 2 -> 5 -> 7   log-probs -1.22, -0.01, -0.67   total -1.90

    The second path wins on the total while losing on both the first and the last
    step, so it is found only by ranking beams on their CUMULATIVE score. Greedy,
    and any scoring that looks at one step at a time, ends up on the first path.
    """
    t = torch.zeros(VOCAB, VOCAB)               # unlisted rows are flat: no strong move
    t[0] = -6.0
    t[0, 1], t[0, 2], t[0, 3] = 3.0, 2.6, 2.5   # start: 1 looks best, 2 is the real win
    t[1] = -0.571
    t[1, 4] = 0.0                               # token 1 -> 4, but only just
    t[2] = -6.0
    t[2, 5] = 0.5                               # token 2 -> 5, nearly certain
    t[4] = -6.0
    t[4, 7] = 0.0                               # 4 -> 7, nearly certain
    t[5] = -2.0
    t[5, 7] = 0.0                               # 5 -> 7, likely
    g = torch.Generator().manual_seed(1234)
    return t + torch.rand(VOCAB, VOCAB, generator=g) * 0.01  # break every tie


TABLE = _table()
LOGP = torch.log_softmax(TABLE, dim=-1)


class _TableLM(torch.nn.Module):
    """model(token_id) -> (1, vocab_size) logits, like the notebook's DummyLM."""

    def __init__(self):
        super().__init__()
        self.vocab_size = VOCAB

    def forward(self, token_id):
        if torch.is_tensor(token_id):
            idx = int(token_id.reshape(-1)[-1].item())
        elif isinstance(token_id, (list, tuple)):
            idx = int(token_id[-1])
        else:
            idx = int(token_id)
        return TABLE[idx].unsqueeze(0)


def _as_ints(seq, where):
    assert seq is not None, f"{where}: beam_search returned None"
    try:
        out = [int(t) for t in seq]
    except TypeError:
        raise AssertionError(
            f"{where}: expected a list of token ids, got {type(seq).__name__}")
    return out


def _score(seq):
    """Cumulative log-probability of a decoded sequence under the test model."""
    return sum(LOGP[seq[i], seq[i + 1]].item() for i in range(len(seq) - 1))


def _brute_force_best(max_len):
    """The true argmax over every sequence — an oracle, not a beam search."""
    best, best_s = None, float("-inf")
    stack = [([0], 0.0)]
    while stack:
        seq, s = stack.pop()
        if len(seq) == max_len:
            if s > best_s:
                best, best_s = seq, s
            continue
        for nxt in range(VOCAB):
            stack.append((seq + [nxt], s + LOGP[seq[-1], nxt].item()))
    return best, best_s


def check_shape_and_start_token(ns):
    model = _TableLM()
    for bw in (1, 2, 4):
        for max_len in (2, 5, 8):
            got = _as_ints(ns.beam_search(model, 0, beam_width=bw, max_len=max_len),
                           f"beam_width={bw}, max_len={max_len}")
            assert len(got) == max_len, (
                f"beam_width={bw}: expected a sequence of length max_len={max_len} "
                f"(the start token counts), got length {len(got)}")
            assert got[0] == 0, \
                f"beam_width={bw}: the first token must be start_token=0, got {got[0]}"
            assert all(0 <= t < VOCAB for t in got), \
                f"decoded token ids outside [0, {VOCAB}): {got}"


def check_beam_width_one_is_greedy(ns):
    """With one beam there is nothing to prune, so it must be argmax decoding."""
    model = _TableLM()
    greedy = [0]
    for _ in range(MAX_LEN - 1):
        greedy.append(int(TABLE[greedy[-1]].argmax()))
    got = _as_ints(ns.beam_search(model, 0, beam_width=1, max_len=MAX_LEN), "beam_width=1")
    assert got == greedy, (
        f"beam_width=1 must reproduce greedy decoding: expected {greedy}, got {got}")


def check_wider_beam_finds_the_best_sequence(ns):
    """The defining property: keep enough beams and you find the best-scoring path."""
    model = _TableLM()
    best, best_score = _brute_force_best(MAX_LEN)
    greedy_score = _score(_as_ints(
        ns.beam_search(model, 0, beam_width=1, max_len=MAX_LEN), "beam_width=1"))
    for bw in (2, 3, 4):
        got = _as_ints(ns.beam_search(model, 0, beam_width=bw, max_len=MAX_LEN),
                       f"beam_width={bw}")
        got_score = _score(got)
        assert abs(got_score - best_score) < 1e-4, (
            f"beam_width={bw} returned {got} scoring {got_score:.3f}; the best "
            f"sequence is {best} scoring {best_score:.3f}. Greedy scores "
            f"{greedy_score:.3f} here, so a beam that big should not settle for "
            f"the greedy path — are candidates scored by their cumulative log-prob?")


def check_returns_the_top_scoring_beam(ns):
    """Different beam widths explore differently but must never score worse than greedy."""
    model = _TableLM()
    greedy_score = _score(_as_ints(
        ns.beam_search(model, 0, beam_width=1, max_len=6), "beam_width=1"))
    for bw in (2, 4):
        got = _as_ints(ns.beam_search(model, 0, beam_width=bw, max_len=6),
                       f"beam_width={bw}")
        s = _score(got)
        assert s >= greedy_score - 1e-4, (
            f"beam_width={bw} returned a sequence scoring {s:.3f}, worse than "
            f"greedy's {greedy_score:.3f} — the final answer should be the beam "
            f"with the highest cumulative score")


def check_deterministic(ns):
    model = _TableLM()
    a = _as_ints(ns.beam_search(model, 0, beam_width=3, max_len=MAX_LEN), "run 1")
    b = _as_ints(ns.beam_search(model, 0, beam_width=3, max_len=MAX_LEN), "run 2")
    assert a == b, (
        f"beam search is deterministic, but two identical calls returned {a} and {b} "
        f"— are you sampling instead of taking the top-k?")


CHECKS = [
    check_shape_and_start_token,
    check_beam_width_one_is_greedy,
    check_wider_beam_finds_the_best_sequence,
    check_returns_the_top_scoring_beam,
    check_deterministic,
]
