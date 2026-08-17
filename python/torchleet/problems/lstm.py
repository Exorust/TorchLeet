"""lstm — the LSTM cell written out by hand, plus the nn.LSTM version.

The hand-rolled cell is not compared against a reference: gate order, parameter
layout and initialisation are all free. Instead it is driven two ways that must
agree — one pass over a whole sequence versus two passes with the returned
(H, C) threaded between them. Only a genuine recurrence with a correctly carried
state satisfies that. The gate algebra is pinned separately by the bound
|H| = |o * tanh(C)| < 1, which fails the moment a sigmoid or tanh is dropped.
"""
import torch
import torch.nn as nn
from torchleet.checks import common as c

ENTRIES = ["CustomLSTMModel", "LSTMModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "CustomLSTMModel(input_dim, hidden_units).forward(inputs, H_C=None) returns "
    "(predictions, (H, C)) with predictions shaped (batch, seq_len, 1).",
    "Per timestep: i, f, o = sigmoid(...), C~ = tanh(...), C = f*C + i*C~, "
    "H = o*tanh(C). Stack every H and push the stack through the Linear.",
    "When H_C is given, start from it instead of a fresh state — that is what "
    "makes the model resumable, and zeros (not randn) are the right default.",
]

FEAT, HID, SEQ = 1, 8, 6


def _custom(ns, seed=0, input_dim=FEAT, hidden=HID):
    torch.manual_seed(seed)
    return ns.CustomLSTMModel(input_dim, hidden)


def _zero_state(batch, hidden=HID):
    return torch.zeros(batch, hidden), torch.zeros(batch, hidden)


def _unpack(out, where):
    assert isinstance(out, (tuple, list)) and len(out) == 2, (
        f"{where}: forward should return (predictions, (H, C)), got "
        f"{type(out).__name__}")
    pred, state = out
    assert isinstance(state, (tuple, list)) and len(state) == 2, (
        f"{where}: the second return value should be the (H, C) pair, got "
        f"{type(state).__name__}")
    return pred, state


def check_custom_output_shape(ns):
    m = _custom(ns)
    m.eval()
    pred, (h, cell) = _unpack(m(torch.randn(3, SEQ, FEAT), _zero_state(3)), "CustomLSTMModel")
    assert tuple(pred.shape) == (3, SEQ, 1), (
        f"expected predictions of shape {(3, SEQ, 1)} — one value per timestep — "
        f"got {tuple(pred.shape)}")
    for name, t in (("H", h), ("C", cell)):
        assert tuple(t.shape) == (3, HID), (
            f"{name} should be (batch, hidden_units) = {(3, HID)}, got "
            f"{tuple(t.shape)}")


def check_custom_honours_its_hidden_size(ns):
    """hidden_units is a constructor argument, not a constant."""
    for hidden in (4, 16):
        m = _custom(ns, hidden=hidden)
        m.eval()
        _, (h, _) = _unpack(m(torch.randn(2, SEQ, FEAT), _zero_state(2, hidden)),
                            f"CustomLSTMModel(1, {hidden})")
        assert tuple(h.shape) == (2, hidden), (
            f"CustomLSTMModel(1, {hidden}) returned a hidden state of "
            f"{tuple(h.shape)}, expected {(2, hidden)}")


def check_custom_state_is_resumable(ns):
    """One pass over 6 steps must equal two passes of 3 with (H, C) threaded."""
    m = _custom(ns)
    m.eval()
    x = torch.randn(2, SEQ, FEAT)
    s0 = _zero_state(2)
    with torch.no_grad():
        full, (hf, cf) = _unpack(m(x, s0), "CustomLSTMModel")
        part1, s1 = _unpack(m(x[:, :3], s0), "CustomLSTMModel")
        part2, (hp, cp) = _unpack(m(x[:, 3:], tuple(s1)), "CustomLSTMModel")
    assert torch.allclose(full[:, :3], part1, atol=1e-5), (
        "the first 3 timesteps changed when the sequence was cut short — a "
        "timestep must only depend on the ones before it")
    assert torch.allclose(full[:, 3:], part2, atol=1e-5), (
        f"feeding steps 4-6 with the state returned after step 3 gave different "
        f"predictions than one pass over all {SEQ} steps (max diff "
        f"{(full[:, 3:] - part2).abs().max():.2e}) — the incoming H_C is being "
        "ignored or the state is not carried forward")
    assert torch.allclose(hf, hp, atol=1e-5) and torch.allclose(cf, cp, atol=1e-5), \
        "the final (H, C) differs between the one-pass and the resumed run"


def check_custom_hidden_state_is_bounded(ns):
    """H = o * tanh(C) with o in (0, 1), so |H| < 1 whatever the input is."""
    m = _custom(ns)
    m.eval()
    with torch.no_grad():
        _, (h, _) = _unpack(m(torch.randn(4, SEQ, FEAT) * 50, _zero_state(4)),
                            "CustomLSTMModel")
    assert float(h.abs().max()) <= 1.0 + 1e-4, (
        f"the hidden state reached {float(h.abs().max()):.3f}; H = o * tanh(C) "
        "with a sigmoid gate can never leave (-1, 1), so a squashing "
        "nonlinearity is missing")


def check_custom_gradients_reach_all_parameters(ns):
    m = _custom(ns)
    pred, _ = _unpack(m(torch.randn(2, SEQ, FEAT), _zero_state(2)), "CustomLSTMModel")
    pred.sum().backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — every gate weight should be on the "
        "forward path (and registered as an nn.Parameter)")


def check_custom_can_learn(ns):
    torch.manual_seed(0)
    m = _custom(ns)
    x = torch.randn(8, SEQ, FEAT)
    y = torch.randn(8, 1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    lossf = nn.MSELoss()

    def step():
        pred, _ = _unpack(m(x, _zero_state(8)), "CustomLSTMModel")
        return lossf(pred[:, -1, :], y)

    first = step().item()
    loss = None
    for _ in range(60):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
    assert loss.item() < first, (
        f"loss did not decrease over 60 steps on a fixed batch "
        f"({first:.4f} -> {loss.item():.4f}) — the custom cell is not learning")


def check_inbuilt_output_shape(ns):
    torch.manual_seed(0)
    m = ns.LSTMModel()
    m.eval()
    out = m(torch.randn(5, SEQ, FEAT))
    assert tuple(out.shape) == (5, 1), (
        f"LSTMModel should return one prediction per sequence — {(5, 1)} for a "
        f"(5, {SEQ}, {FEAT}) input — got {tuple(out.shape)}")


def check_inbuilt_any_sequence_length(ns):
    torch.manual_seed(0)
    m = ns.LSTMModel()
    m.eval()
    for t in (3, 10, 25):
        out = m(torch.randn(2, t, FEAT))
        assert tuple(out.shape) == (2, 1), (
            f"sequence length {t} gave {tuple(out.shape)}, expected {(2, 1)} — "
            "an LSTM accepts any number of timesteps")


CHECKS = [
    check_custom_output_shape,
    check_custom_honours_its_hidden_size,
    check_custom_state_is_resumable,
    check_custom_hidden_state_is_bounded,
    check_custom_gradients_reach_all_parameters,
    check_custom_can_learn,
    check_inbuilt_output_shape,
    check_inbuilt_any_sequence_length,
    c.gradients_flow("LSTMModel", (SEQ, FEAT)),
    c.can_learn("LSTMModel", (SEQ, FEAT), (1,), steps=60, lr=1e-2),
]
