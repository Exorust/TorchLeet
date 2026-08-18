"""inference-engine — tokenizer, KV cache, a mini transformer, nucleus sampling
and the serving loop that ties them together.

The spine of this spec is one property applied at three levels: decoding a
sequence token by token *with* the KV cache must give exactly the same numbers
as one full forward pass over the whole sequence. It is checked on
MultiHeadAttention, on TransformerBlock and on MiniTransformer (where it also
pins the positional offset the cache implies). Both runs use the solver's own
module, so weight init and op order can never produce a false failure — only a
genuinely wrong cache can. Causality is checked the same way: poke the last
token and nothing earlier may move.
"""
import torch

ENTRIES = ["SimpleTokenizer", "KVCache", "MultiHeadAttention", "FeedForward",
           "TransformerBlock", "MiniTransformer", "top_p_sample", "InferenceEngine"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "The cache stores K and V as (B, heads, seq_len, d_head) and update() "
    "concatenates the new step along dim=2. In cached mode you only project the "
    "NEW token into q/k/v and attend over the whole cached K/V.",
    "MiniTransformer has to know where it is: when a cache is passed, the "
    "positional embedding index for the incoming token is the cache's current "
    "seq_len, not 0.",
    "top_p keeps the smallest set of tokens whose cumulative probability reaches "
    "top_p: sort descending, cumsum, zero out the tail, RENORMALISE, sample, then "
    "map the sampled index back through sorted_indices.",
]

V, D_MODEL, HEADS, D_FF, LAYERS, MAXLEN = 40, 32, 4, 64, 2, 64
D_HEAD = D_MODEL // HEADS
B, S = 2, 9


def _mha(ns):
    torch.manual_seed(0)
    m = ns.MultiHeadAttention(D_MODEL, HEADS)
    m.eval()
    return m


def _block(ns):
    torch.manual_seed(1)
    m = ns.TransformerBlock(D_MODEL, HEADS, D_FF)
    m.eval()
    return m


def _model(ns, vocab=V):
    torch.manual_seed(2)
    m = ns.MiniTransformer(vocab_size=vocab, d_model=D_MODEL, num_heads=HEADS,
                           d_ff=D_FF, num_layers=LAYERS, max_seq_len=MAXLEN)
    m.eval()
    return m


def _n_blocks(ns, model):
    return sum(1 for mod in model.modules() if isinstance(mod, ns.TransformerBlock))


def _bump_last(x):
    """A perturbation of the final token that survives a LayerNorm."""
    x2 = x.clone()
    x2[:, -1, :] = x[:, -1, :] * 2.0 + torch.linspace(-3.0, 3.0, x.shape[-1])
    return x2


# --------------------------------------------------------------------------
# SimpleTokenizer
# --------------------------------------------------------------------------
def check_tokenizer_round_trips_text(ns):
    tok = ns.SimpleTokenizer()
    for text in ("Hello, World!", "def fibonacci(n):", "a b\tc", ""):
        ids = tok.encode(text)
        assert isinstance(ids, (list, tuple)), \
            f"encode should return a sequence of ids, got {type(ids).__name__}"
        assert all(isinstance(int(i), int) for i in ids), "encode must return integer ids"
        assert tok.decode(ids) == text, \
            f"decode(encode({text!r})) should give back {text!r}, got {tok.decode(ids)!r}"


def check_tokenizer_ids_fit_the_vocabulary(ns):
    tok = ns.SimpleTokenizer()
    text = "".join(chr(c) for c in range(32, 127))
    ids = [int(i) for i in tok.encode(text)]
    assert max(ids) < int(tok.vocab_size), (
        f"encode produced id {max(ids)} but vocab_size is {tok.vocab_size}; the "
        f"output projection would index out of range")
    assert min(ids) > max(int(tok.pad_id), int(tok.eos_id)), (
        f"regular characters must not collide with the special ids "
        f"(pad={tok.pad_id}, eos={tok.eos_id}); the smallest character id is {min(ids)}")


def check_tokenizer_drops_special_tokens_when_decoding(ns):
    tok = ns.SimpleTokenizer()
    assert int(tok.pad_id) != int(tok.eos_id), \
        "pad and eos need distinct ids so the engine can tell padding from a stop"
    ids = [int(i) for i in tok.encode("hi")]
    noisy = [int(tok.pad_id)] + ids + [int(tok.eos_id), int(tok.pad_id)]
    assert tok.decode(noisy) == "hi", (
        "decode must skip pad and eos — they are control ids, not characters; got "
        f"{tok.decode(noisy)!r}")


# --------------------------------------------------------------------------
# KVCache
# --------------------------------------------------------------------------
def check_cache_accumulates_along_the_sequence_axis(ns):
    c = ns.KVCache()
    ks = [torch.randn(B, HEADS, 1, D_HEAD) for _ in range(4)]
    vs = [torch.randn(B, HEADS, 1, D_HEAD) for _ in range(4)]
    fk, fv = c.update(ks[0], vs[0])
    assert tuple(fk.shape) == (B, HEADS, 1, D_HEAD), (
        f"the first update returns just the new K, {(B, HEADS, 1, D_HEAD)}; "
        f"got {tuple(fk.shape)}")
    for k, v in zip(ks[1:], vs[1:]):
        fk, fv = c.update(k, v)
    assert tuple(fk.shape) == (B, HEADS, 4, D_HEAD), (
        f"after 4 single-token updates the cache holds 4 positions, "
        f"{(B, HEADS, 4, D_HEAD)}; got {tuple(fk.shape)} — K/V are appended along "
        f"dim=2, the sequence axis")
    assert torch.allclose(fk, torch.cat(ks, dim=2), atol=1e-6), \
        "the cached K is not the past keys concatenated in decode order"
    assert torch.allclose(fv, torch.cat(vs, dim=2), atol=1e-6), \
        "the cached V is not the past values concatenated in decode order"


def check_cache_reports_its_length_and_can_be_reset(ns):
    c = ns.KVCache()
    assert int(getattr(c, "seq_len", 0)) == 0, (
        f"a fresh cache holds nothing, so seq_len should be 0; got "
        f"{getattr(c, 'seq_len', 'no attribute')}. MiniTransformer needs it to "
        f"know which position the incoming token is at.")
    for step in range(3):
        k = torch.randn(B, HEADS, 1, D_HEAD)
        c.update(k, k)
        assert int(c.seq_len) == step + 1, \
            f"after {step + 1} updates seq_len should be {step + 1}, got {int(c.seq_len)}"
    if not hasattr(c, "reset"):
        from torchleet.runner import Skip
        raise Skip("no reset() on this KVCache")
    c.reset()
    assert int(c.seq_len) == 0, f"after reset() the cache must be empty, seq_len is {int(c.seq_len)}"
    k = torch.randn(B, HEADS, 1, D_HEAD)
    fk, _ = c.update(k, k)
    assert tuple(fk.shape) == tuple(k.shape), \
        f"after reset() the next update starts over; got K {tuple(fk.shape)}"


# --------------------------------------------------------------------------
# MultiHeadAttention
# --------------------------------------------------------------------------
def check_attention_shape(ns):
    out = _mha(ns)(torch.randn(B, S, D_MODEL))
    assert tuple(out.shape) == (B, S, D_MODEL), \
        f"attention must preserve (B, S, d_model) = {(B, S, D_MODEL)}, got {tuple(out.shape)}"


def check_attention_is_causal(ns):
    m = _mha(ns)
    x = torch.randn(B, S, D_MODEL)
    with torch.no_grad():
        a, b = m(x), m(_bump_last(x))
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6), (
        "changing the LAST token changed the output at earlier positions — "
        "full-sequence attention needs a causal mask so position t only sees "
        f"0..t; max deviation {float((a[:, :-1] - b[:, :-1]).abs().max()):.6f}")


def check_cached_attention_matches_full_recomputation(ns):
    m = _mha(ns)
    x = torch.randn(B, S, D_MODEL)
    with torch.no_grad():
        full = m(x, kv_cache=None)
        cache = ns.KVCache()
        steps = []
        for t in range(S):
            out_t = m(x[:, t:t + 1, :], kv_cache=cache)
            assert tuple(out_t.shape) == (B, 1, D_MODEL), (
                f"cached step {t} should return one token's output "
                f"{(B, 1, D_MODEL)}, got {tuple(out_t.shape)}")
            steps.append(out_t)
    cached = torch.cat(steps, dim=1)
    assert torch.allclose(full, cached, atol=1e-5), (
        "decoding token by token through the KV cache disagrees with one full "
        f"forward pass (max diff {float((full - cached).abs().max()):.2e}). In "
        "cached mode project only the new token into q/k/v, then attend over the "
        "whole cached K/V — and do not re-apply a causal mask, the cache already "
        "holds only the past.")


# --------------------------------------------------------------------------
# FeedForward
# --------------------------------------------------------------------------
def check_feedforward_preserves_d_model(ns):
    ff = ns.FeedForward(D_MODEL, D_FF)
    for shape in ((7, D_MODEL), (B, S, D_MODEL)):
        out = ff(torch.randn(*shape))
        assert tuple(out.shape) == shape, \
            f"the FFN maps d_model back to d_model: expected {shape}, got {tuple(out.shape)}"
    inner = [p for p in ff.parameters() if p.dim() == 2 and D_FF in p.shape]
    assert inner, (
        f"the FFN should widen to d_ff={D_FF} in the middle; no weight matrix "
        f"touching {D_FF} was found")


def check_feedforward_is_nonlinear(ns):
    ff = ns.FeedForward(D_MODEL, D_FF)
    ff.eval()
    torch.manual_seed(0)
    a, b = torch.randn(4, D_MODEL), torch.randn(4, D_MODEL)
    with torch.no_grad():
        gap = ff(a + b) - ff(a) - ff(b) + ff(torch.zeros(4, D_MODEL))
    assert float(gap.abs().max()) > 1e-4, (
        "f(a+b) - f(a) - f(b) + f(0) came out 0, so the FFN is affine — two "
        "Linears stacked without an activation between them collapse to one "
        "Linear. Put GELU/ReLU in the middle.")


def check_feedforward_gradients_flow(ns):
    ff = ns.FeedForward(D_MODEL, D_FF)
    x = torch.randn(B, S, D_MODEL, requires_grad=True)
    ff(x).pow(2).sum().backward()
    assert x.grad is not None and bool((x.grad != 0).any()), \
        "no gradient reached the FFN input"


# --------------------------------------------------------------------------
# TransformerBlock
# --------------------------------------------------------------------------
def check_block_shape_and_causality(ns):
    blk = _block(ns)
    x = torch.randn(B, S, D_MODEL)
    with torch.no_grad():
        a, b = blk(x), blk(_bump_last(x))
    assert tuple(a.shape) == (B, S, D_MODEL), \
        f"the block preserves (B, S, d_model) = {(B, S, D_MODEL)}, got {tuple(a.shape)}"
    assert torch.isfinite(a).all(), "block output contains NaN/Inf"
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6), (
        "perturbing the last token moved earlier positions — the block must stay "
        "causal (only the attention sublayer mixes positions, and it is masked)")


def check_block_cached_decode_matches_full(ns):
    blk = _block(ns)
    x = torch.randn(B, S, D_MODEL)
    with torch.no_grad():
        full = blk(x)
        cache = ns.KVCache()
        cached = torch.cat([blk(x[:, t:t + 1, :], kv_cache=cache) for t in range(S)], dim=1)
    assert torch.allclose(full, cached, atol=1e-5), (
        "stepping the block one token at a time with a KV cache disagrees with the "
        f"full forward (max diff {float((full - cached).abs().max()):.2e}); the "
        "cache has to be threaded into the attention sublayer")


def check_block_is_residual(ns):
    """A pre-norm block adds its sublayers to the input, so the input must still
    be visible in the output."""
    blk = _block(ns)
    x = torch.randn(1, 1, D_MODEL) * 8.0
    with torch.no_grad():
        out = blk(x)
    assert float((out - x).norm()) < float(x.norm()), (
        "with a large input the output drifted further from x than x itself — a "
        "pre-norm block computes x + attn(ln1(x)) then x + ffn(ln2(x)), so the "
        "residual path must carry the input through")


# --------------------------------------------------------------------------
# MiniTransformer
# --------------------------------------------------------------------------
def check_transformer_logits_shape_and_depth(ns):
    m = _model(ns)
    ids = torch.randint(0, V, (B, S))
    with torch.no_grad():
        logits = m(ids)
    assert tuple(logits.shape) == (B, S, V), (
        f"the model scores every position over the vocabulary, {(B, S, V)}; "
        f"got {tuple(logits.shape)}")
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
    n = _n_blocks(ns, m)
    assert n == LAYERS, \
        f"num_layers={LAYERS} was requested but the model built {n} TransformerBlock(s)"


def check_transformer_is_causal(ns):
    m = _model(ns)
    ids = torch.randint(0, V, (B, S))
    other = ids.clone()
    other[:, -1] = (other[:, -1] + 7) % V
    with torch.no_grad():
        a, b = m(ids), m(other)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5), (
        "changing the last token changed the logits at earlier positions — the "
        "stack is not causal, so training and cached decoding would disagree")


def check_transformer_is_position_aware(ns):
    m = _model(ns)
    ids = torch.tensor([[3, 3, 3, 3]])
    with torch.no_grad():
        logits = m(ids)
    assert not torch.allclose(logits[0, 0], logits[0, 1], atol=1e-6), (
        "four copies of the same token gave identical logits at every position — "
        "the positional embedding is not being added")


def check_cached_generation_matches_full_recomputation(ns):
    """The property the whole engine rests on."""
    m = _model(ns)
    ids = torch.randint(0, V, (B, S))
    n = _n_blocks(ns, m) or LAYERS
    with torch.no_grad():
        full = m(ids)
        caches = [ns.KVCache() for _ in range(n)]
        steps = []
        for t in range(S):
            out_t = m(ids[:, t:t + 1], kv_caches=caches)
            assert tuple(out_t.shape) == (B, 1, V), (
                f"decode step {t} should return logits for one token {(B, 1, V)}, "
                f"got {tuple(out_t.shape)}")
            steps.append(out_t)
    cached = torch.cat(steps, dim=1)
    assert torch.allclose(full, cached, atol=1e-4), (
        "generating token by token with KV caches does not reproduce a single "
        f"full forward pass (max diff {float((full - cached).abs().max()):.2e}). "
        "The usual cause is the positional embedding: with a cache the incoming "
        "token sits at index cache.seq_len, not at 0.")


# --------------------------------------------------------------------------
# top_p_sample
# --------------------------------------------------------------------------
def check_top_p_output_shape(ns):
    out = ns.top_p_sample(torch.randn(5, V), temperature=1.0, top_p=0.9)
    assert tuple(out.shape) == (5, 1), \
        f"top_p_sample returns one token id per row, (B, 1) = {(5, 1)}; got {tuple(out.shape)}"
    assert out.dtype in (torch.long, torch.int64, torch.int32), \
        f"sampled ids must be integers, got {out.dtype}"
    assert int(out.min()) >= 0 and int(out.max()) < V, \
        f"sampled ids must index the vocabulary [0, {V}), got [{int(out.min())}, {int(out.max())}]"


def check_low_temperature_collapses_to_the_argmax(ns):
    torch.manual_seed(0)
    logits = torch.randperm(V).float().unsqueeze(0)
    want = int(logits.argmax())
    got = {int(ns.top_p_sample(logits, temperature=0.01, top_p=1.0)) for _ in range(50)}
    assert got == {want}, (
        f"at temperature 0.01 the distribution is effectively one-hot on the "
        f"largest logit ({want}), so sampling must always return it; got {sorted(got)}. "
        f"The temperature has to DIVIDE the logits before the softmax.")


def check_top_p_drops_the_tail(ns):
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    torch.manual_seed(0)
    seen = {int(ns.top_p_sample(logits, temperature=1.0, top_p=0.9)) for _ in range(400)}
    assert 3 not in seen, (
        "with probabilities [0.5, 0.3, 0.15, 0.05] and top_p=0.9 the nucleus is "
        "reached before the last token, so token 3 must never be sampled; it was")
    assert {0, 1} <= seen, (
        f"tokens inside the nucleus must all remain reachable after renormalising; "
        f"only {sorted(seen)} were ever sampled in 400 draws")


def check_top_p_keeps_a_dominant_token(ns):
    logits = torch.zeros(1, V)
    logits[0, 11] = 30.0
    got = {int(ns.top_p_sample(logits, temperature=1.0, top_p=0.9)) for _ in range(100)}
    assert got == {11}, (
        "one token already carries essentially all the probability mass, so the "
        f"nucleus is just that token; got {sorted(got)}")


# --------------------------------------------------------------------------
# InferenceEngine
# --------------------------------------------------------------------------
PROMPTS = ["ab", "hello there", "x"]


def _engine(ns):
    tok = ns.SimpleTokenizer()
    model = _model(ns, vocab=int(tok.vocab_size))
    return ns.InferenceEngine(model, tok, device="cpu")


def check_engine_completes_every_prompt(ns):
    eng = _engine(ns)
    with torch.no_grad():
        out = eng.generate(PROMPTS, max_new_tokens=10, temperature=1.0, top_p=0.9)
    assert isinstance(out, (list, tuple)) and len(out) == len(PROMPTS), (
        f"generate should return one completion per prompt ({len(PROMPTS)}), got "
        f"{len(out) if hasattr(out, '__len__') else type(out).__name__}")
    for prompt, text in zip(PROMPTS, out):
        assert isinstance(text, str), f"each completion should be a decoded string, got {type(text).__name__}"
        assert text.startswith(prompt), (
            f"the returned text should be prompt + completion; {text!r} does not "
            f"start with {prompt!r}. Note the prompts have different lengths, so "
            f"the padding has to be undone before decoding.")
        assert len(text) <= len(prompt) + 10, (
            f"asked for at most 10 new tokens but got {len(text) - len(prompt)} "
            f"extra characters for prompt {prompt!r}")
    assert any(len(t) > len(p) for p, t in zip(PROMPTS, out)), \
        "no prompt produced a single new token in 10 steps"


def check_engine_respects_max_new_tokens(ns):
    eng = _engine(ns)
    with torch.no_grad():
        out = eng.generate(PROMPTS, max_new_tokens=0, temperature=1.0, top_p=0.9)
    assert list(out) == PROMPTS, (
        f"max_new_tokens=0 means nothing is generated, so the prompts come back "
        f"unchanged; got {list(out)}")


def check_engine_is_deterministic_when_sampling_is(ns):
    eng = _engine(ns)
    with torch.no_grad():
        a = eng.generate(PROMPTS, max_new_tokens=6, temperature=1e-4, top_p=1.0)
        b = eng.generate(PROMPTS, max_new_tokens=6, temperature=1e-4, top_p=1.0)
    assert list(a) == list(b), (
        "at temperature 1e-4 sampling is effectively greedy, so two calls must "
        f"produce the same text; got {list(a)} then {list(b)}. Leftover state "
        "between calls (a KV cache that is not recreated per request) does this.")


CHECKS = [
    check_tokenizer_round_trips_text,
    check_tokenizer_ids_fit_the_vocabulary,
    check_tokenizer_drops_special_tokens_when_decoding,
    check_cache_accumulates_along_the_sequence_axis,
    check_cache_reports_its_length_and_can_be_reset,
    check_attention_shape,
    check_attention_is_causal,
    check_cached_attention_matches_full_recomputation,
    check_feedforward_preserves_d_model,
    check_feedforward_is_nonlinear,
    check_feedforward_gradients_flow,
    check_block_shape_and_causality,
    check_block_cached_decode_matches_full,
    check_block_is_residual,
    check_transformer_logits_shape_and_depth,
    check_transformer_is_causal,
    check_transformer_is_position_aware,
    check_cached_generation_matches_full_recomputation,
    check_top_p_output_shape,
    check_low_temperature_collapses_to_the_argmax,
    check_top_p_drops_the_tail,
    check_top_p_keeps_a_dominant_token,
    check_engine_completes_every_prompt,
    check_engine_respects_max_new_tokens,
    check_engine_is_deterministic_when_sampling_is,
]
