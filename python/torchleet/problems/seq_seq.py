"""seq-seq — the encoder half of an attention seq2seq model.

The decoder consumes three things from the encoder: per-timestep outputs to
attend over, and the final (hidden, cell) pair to start decoding from. These
checks pin exactly that interface and then use the one invariant that ties the
two return values together — for a left-to-right LSTM the last row of `outputs`
*is* the top layer's final hidden state. An encoder that returns the states of
the wrong layer, forgets batch_first, or hands back the embeddings instead of
the LSTM outputs fails it.
"""
import torch
import torch.nn as nn

ENTRIES = ["Encoder"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Encoder(input_dim, embed_dim, hidden_dim, num_layers): nn.Embedding over the "
    "source vocabulary, then nn.LSTM(embed_dim, hidden_dim, num_layers, "
    "batch_first=True).",
    "forward returns (outputs, (hidden, cell)) — the attention needs every "
    "timestep, the decoder needs the final states to start from.",
    "With batch_first=True outputs is (batch, src_len, hidden_dim) while hidden "
    "and cell stay (num_layers, batch, hidden_dim).",
]

VOCAB, EMBED, HIDDEN, LAYERS, SRC = 20, 12, 16, 2, 7


def _encoder(ns, seed=0, vocab=VOCAB, embed=EMBED, hidden=HIDDEN, layers=LAYERS):
    torch.manual_seed(seed)
    return ns.Encoder(vocab, embed, hidden, layers)


def _run(enc, batch=3, src=SRC, vocab=VOCAB):
    x = torch.randint(0, vocab, (batch, src))
    out = enc(x)
    assert isinstance(out, (tuple, list)) and len(out) == 2, (
        "Encoder.forward should return (outputs, (hidden, cell)), got "
        f"{type(out).__name__}"
        + (f" of length {len(out)}" if hasattr(out, "__len__") else ""))
    outputs, state = out
    assert isinstance(state, (tuple, list)) and len(state) == 2, (
        "the second return value should be the (hidden, cell) pair from the "
        f"LSTM, got {type(state).__name__}")
    return outputs, state[0], state[1]


def check_returns_outputs_and_states(ns):
    enc = _encoder(ns)
    enc.eval()
    outputs, hidden, cell = _run(enc)
    assert tuple(outputs.shape) == (3, SRC, HIDDEN), (
        f"outputs should be (batch, src_len, hidden_dim) = {(3, SRC, HIDDEN)} so "
        f"attention can score every source position, got {tuple(outputs.shape)}"
        + (" — pass batch_first=True to nn.LSTM"
           if tuple(outputs.shape) == (SRC, 3, HIDDEN) else ""))
    for name, t in (("hidden", hidden), ("cell", cell)):
        assert tuple(t.shape) == (LAYERS, 3, HIDDEN), (
            f"{name} should be (num_layers, batch, hidden_dim) = "
            f"{(LAYERS, 3, HIDDEN)}, got {tuple(t.shape)}")


def check_last_output_is_the_final_hidden_state(ns):
    """A left-to-right LSTM ends its last step in exactly h_n[-1]."""
    enc = _encoder(ns)
    enc.eval()
    with torch.no_grad():
        outputs, hidden, _ = _run(enc)
    assert torch.allclose(outputs[:, -1, :], hidden[-1], atol=1e-5), (
        "outputs[:, -1, :] does not equal hidden[-1] (max diff "
        f"{(outputs[:, -1, :] - hidden[-1]).abs().max():.2e}) — the two return "
        "values do not come from the same single-direction LSTM pass, or the "
        "batch and time axes are swapped")


def check_accepts_any_batch_and_source_length(ns):
    enc = _encoder(ns)
    enc.eval()
    for batch, src in ((1, 4), (5, 7), (2, 15)):
        outputs, hidden, cell = _run(enc, batch=batch, src=src)
        assert tuple(outputs.shape) == (batch, src, HIDDEN), (
            f"a ({batch}, {src}) source batch gave outputs {tuple(outputs.shape)}, "
            f"expected {(batch, src, HIDDEN)}")
        assert tuple(hidden.shape) == (LAYERS, batch, HIDDEN), \
            f"hidden was {tuple(hidden.shape)}, expected {(LAYERS, batch, HIDDEN)}"


def check_honours_its_constructor_arguments(ns):
    """hidden_dim and num_layers are arguments, not constants."""
    enc = _encoder(ns, hidden=32, layers=1)
    enc.eval()
    outputs, hidden, _ = _run(enc)
    assert outputs.shape[-1] == 32, (
        f"Encoder(..., hidden_dim=32, num_layers=1) produced outputs with "
        f"{outputs.shape[-1]} features, expected 32")
    assert hidden.shape[0] == 1, (
        f"num_layers=1 should give a hidden state with a leading 1, got "
        f"{tuple(hidden.shape)}")


def check_embeds_the_whole_vocabulary(ns):
    """input_dim is the source vocabulary size, so every id below it must work."""
    enc = _encoder(ns)
    enc.eval()
    ids = torch.arange(VOCAB).unsqueeze(0)          # 0 .. VOCAB-1
    with torch.no_grad():
        outputs, _, _ = _run(enc, batch=1, src=VOCAB)   # shape sanity first
        a, _ = enc(ids)
        b, _ = enc(ids.clone())
        changed, _ = enc(torch.cat([ids[:, :1] * 0 + (VOCAB - 1), ids[:, 1:]], dim=1))
    assert torch.allclose(a, b, atol=1e-6), \
        "the same token ids gave two different encodings"
    assert not torch.allclose(a, changed, atol=1e-6), (
        "changing the first source token left the encoding unchanged — the input "
        "ids are not reaching the embedding")


def check_gradients_reach_all_parameters(ns):
    enc = _encoder(ns)
    outputs, hidden, cell = _run(enc, batch=2)
    (outputs.sum() + hidden.sum() + cell.sum()).backward()
    dead = [n for n, p in enc.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — these layers are not on the forward path")


def check_can_learn(ns):
    """Overfit one fixed batch through a tiny read-out head; loss must fall."""
    torch.manual_seed(0)
    enc = _encoder(ns)
    head = nn.Linear(HIDDEN, 1)
    x = torch.randint(0, VOCAB, (8, SRC))
    y = torch.randn(8, 1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-2)
    lossf = nn.MSELoss()

    def step():
        outputs, _ = enc(x)
        return lossf(head(outputs[:, -1, :]), y)

    first = step().item()
    loss = None
    for _ in range(40):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
    assert loss.item() < first, (
        f"loss did not decrease over 40 steps on a fixed batch "
        f"({first:.4f} -> {loss.item():.4f}) — the encoder is not trainable")


CHECKS = [
    check_returns_outputs_and_states,
    check_last_output_is_the_final_hidden_state,
    check_accepts_any_batch_and_source_length,
    check_honours_its_constructor_arguments,
    check_embeds_the_whole_vocabulary,
    check_gradients_reach_all_parameters,
    check_can_learn,
]
