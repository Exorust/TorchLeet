"""quantize-lm — an LSTM language model that survives dynamic quantization.

The model itself is free-form, so the checks pin the contract stated in the
problem (token ids in, one probability distribution over the vocabulary out) and
then verify the thing the exercise is actually about: `quantize_dynamic` can swap
the recurrent/linear layers for int8 ones, the result is materially smaller, and
it still runs. A model built from ops dynamic quantization cannot touch — no
nn.Linear, no nn.LSTM/GRU/RNN — fails that last check.
"""
import io

import torch
import torch.nn as nn
from torchleet.runner import Skip

ENTRIES = ["LanguageModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "LanguageModel(vocab_size, embed_size, hidden_size, num_layers): Embedding -> "
    "LSTM(batch_first=True) -> Linear -> Softmax.",
    "Predict the *next* token, so take the last timestep of the LSTM output "
    "(lstm_out[:, -1, :]) before the Linear — the result is (batch, vocab_size).",
    "torch.ao.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM}, "
    "dtype=torch.qint8) only rewrites modules it recognises, so build the model "
    "out of nn.Linear / nn.LSTM rather than raw matmuls.",
]

VOCAB, EMBED, HIDDEN, LAYERS, SEQ = 50, 32, 48, 2, 8


def _model(ns, seed=0):
    """Built in eval mode — this is an inference-time exercise."""
    torch.manual_seed(seed)
    m = ns.LanguageModel(VOCAB, EMBED, HIDDEN, LAYERS)
    m.eval()
    return m


def _tokens(batch=4, seq=SEQ):
    return torch.randint(0, VOCAB, (batch, seq))


def check_output_shape(ns):
    m = _model(ns)
    out = m(_tokens())
    assert out.ndim == 2, (
        f"expected a 2D (batch, vocab_size) score for the next token, got a "
        f"{out.ndim}D tensor {tuple(out.shape)} — reduce the sequence dimension "
        "by taking the last LSTM timestep")
    assert tuple(out.shape) == (4, VOCAB), \
        f"expected (4, {VOCAB}) for a (4, {SEQ}) batch of token ids, got {tuple(out.shape)}"


def check_batch_and_sequence_agnostic(ns):
    m = _model(ns)
    for b, t in ((1, 3), (3, 8), (7, 20)):
        out = m(_tokens(b, t))
        assert tuple(out.shape) == (b, VOCAB), (
            f"a ({b}, {t}) batch of ids gave {tuple(out.shape)}, expected "
            f"{(b, VOCAB)} — the model must accept any batch and any length")


def check_output_is_a_distribution(ns):
    """The problem asks for a softmax over the vocabulary."""
    m = _model(ns)
    with torch.no_grad():
        out = m(_tokens())
    assert float(out.min()) >= 0.0, (
        f"the smallest output is {float(out.min()):.4f}; a probability over the "
        "vocabulary cannot be negative — apply softmax to the linear output")
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
        f"each row should sum to 1 after the softmax, got "
        f"{[round(float(s), 4) for s in sums[:4]]}")


def check_gradients_reach_all_parameters(ns):
    m = _model(ns)
    out = m(_tokens(2))
    # A random projection, not .sum(): the rows of a softmax always add to 1, so
    # summing them would give every parameter a zero gradient by construction.
    torch.manual_seed(1)
    (out * torch.randn_like(out)).sum().backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — these layers are not on the forward path")


def _state_dict_bytes(module):
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes


def check_dynamic_quantization_shrinks_the_model(ns):
    """The point of the exercise: int8 weights, same interface, smaller file."""
    if torch.backends.quantized.engine == "none":
        engines = [e for e in torch.backends.quantized.supported_engines
                   if e != "none"]
        if not engines:
            raise Skip("no quantized backend (fbgemm/qnnpack) built into this torch")
        torch.backends.quantized.engine = engines[0]

    from torch.ao.quantization import quantize_dynamic

    m = _model(ns)
    m.eval()
    before = _state_dict_bytes(m)
    q = quantize_dynamic(m, {nn.Linear, nn.LSTM, nn.GRU, nn.RNN}, dtype=torch.qint8)
    after = _state_dict_bytes(q)
    assert after < before, (
        f"the quantized model is {after} bytes against {before} for the float "
        "one — quantize_dynamic found nothing it could convert, so the weights "
        "are not in nn.Linear / nn.LSTM modules")

    x = _tokens(2)
    with torch.no_grad():
        out = q(x)
    assert tuple(out.shape) == (2, VOCAB), (
        f"after quantization the model returns {tuple(out.shape)} instead of "
        f"{(2, VOCAB)}")
    assert not torch.isnan(out).any(), "the quantized model produces NaN"


CHECKS = [
    check_output_shape,
    check_batch_and_sequence_agnostic,
    check_output_is_a_distribution,
    check_gradients_reach_all_parameters,
    check_dynamic_quantization_shrinks_the_model,
]
