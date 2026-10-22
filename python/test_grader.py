"""Regression test for the property-based grader.

The invariant that matters: a CORRECT solution written differently from ours must
pass. An earlier design diffed user output against a bundled reference, which
fails that case for anything involving model construction. If this file ever goes
red on a `divergent` case, the grader has regressed to output-diffing.

Run: python3 test_grader.py     (needs torch; no test framework)
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from torchleet import check  # noqa: E402


# ------------------------------------------------------------------ kv-cache
class KVCache:
    def __init__(self):
        self.k = self.v = None

    def update(self, nk, nv):
        self.k = nk if self.k is None else torch.cat([self.k, nk], dim=2)
        self.v = nv if self.v is None else torch.cat([self.v, nv], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = self.v = None


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.d_model, self.num_heads, self.d_head = d, h, d // h
        for n in "qkvo":
            setattr(self, f"W_{n}", nn.Linear(d, d, bias=False))

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        q, k, v = (t.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
                   for t in (self.W_q(x), self.W_k(x), self.W_v(x)))
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if kv_cache is None:
            m = torch.tril(torch.ones(S, S)).bool()
            s = s.masked_fill(~m, float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(o)


class ListKVCache(KVCache):
    """Correct, but stores a list and concatenates on demand."""

    def __init__(self):
        self._k, self._v = [], []

    def update(self, nk, nv):
        self._k.append(nk)
        self._v.append(nv)
        return torch.cat(self._k, dim=2), torch.cat(self._v, dim=2)

    def reset(self):
        self._k, self._v = [], []


class SdpaAttn(Attn):
    """Correct, but xavier init + F.linear + fused SDPA."""

    def __init__(self, d, h):
        super().__init__(d, h)
        for n in "qkvo":
            nn.init.xavier_uniform_(getattr(self, f"W_{n}").weight)

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        q, k, v = (F.linear(x, getattr(self, f"W_{n}").weight)
                   .reshape(B, S, self.num_heads, self.d_head).permute(0, 2, 1, 3)
                   for n in "qkv")
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.permute(0, 2, 1, 3).reshape(B, S, self.d_model)
        return F.linear(o, self.W_o.weight)


class NoPastAttn(Attn):
    """Broken: updates the cache then ignores the past."""

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        q, k, v = (t.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
                   for t in (self.W_q(x), self.W_k(x), self.W_v(x)))
        if kv_cache is not None:
            kv_cache.update(k, v)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if kv_cache is None:
            s = s.masked_fill(~torch.tril(torch.ones(S, S)).bool(), float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(o)


# ------------------------------------------------------------------- softmax
def online_softmax(x):
    m = torch.full((x.shape[0], 1), float("-inf"))
    s = torch.zeros(x.shape[0], 1)
    for j in range(x.shape[1]):
        xj = x[:, j:j + 1]
        mn = torch.maximum(m, xj)
        s = s * torch.exp(m - mn) + torch.exp(xj - mn)
        m = mn
    return torch.exp(x - m) / s


def unstable_softmax(x):
    e = torch.exp(x)
    return e / e.sum(-1, keepdim=True)


# ----------------------------------------------------------------------- cnn
class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(3, 32, 3, 1, 1), nn.Conv2d(32, 64, 3, 1, 1)
        self.p, self.f1, self.f2 = nn.MaxPool2d(2, 2), nn.Linear(64 * 16 * 16, 128), nn.Linear(128, 10)

    def forward(self, x):
        x = self.p(F.relu(self.c2(F.relu(self.c1(x)))))
        return self.f2(F.relu(self.f1(x.flatten(1))))


class DivergentCnn(nn.Module):
    """Correct, but a completely different architecture."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2), nn.BatchNorm2d(16), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(32 * 16, 10))

    def forward(self, x):
        return self.net(x)


class WrongClassCount(Cnn):
    def __init__(self):
        super().__init__()
        self.f2 = nn.Linear(128, 100)


# ---------------------------------------------------------------------- rope
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_apply(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def rope_noop(q, k, cos, sin):
    """Broken: ignores cos/sin and returns the inputs untouched."""
    return q, k


def rope_query_only(q, k, cos, sin):
    """Broken: rotates the query but forgets the key."""
    return q * cos + rotate_half(q) * sin, k


# --------------------------------------------------------- gradient checkpointing
class CkptFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *args):
        ctx.fn = fn
        ctx.save_for_backward(*args)
        with torch.no_grad():
            return fn(*args)

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = [t.detach().requires_grad_(t.requires_grad) for t in ctx.saved_tensors]
        with torch.enable_grad():
            out = ctx.fn(*inputs)
        outs = out if isinstance(out, tuple) else (out,)
        pairs = [(o, g) for o, g in zip(outs, grad_outputs) if o.requires_grad]
        if pairs:
            torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        return (None, *(t.grad if t.requires_grad else None for t in inputs))


def ckpt(fn, *args):
    if torch.is_grad_enabled() and not any(
            isinstance(a, torch.Tensor) and a.requires_grad for a in args):
        dummy = torch.empty(0, requires_grad=True)
        return CkptFn.apply(lambda _d, *a: fn(*a), dummy, *args)
    return CkptFn.apply(fn, *args)


def ckpt_no_dummy(fn, *args):
    """Broken: with a plain (no-grad) input backward never runs, so fn never trains."""
    return CkptFn.apply(fn, *args)


class CkptFnInputsOnly(CkptFn):
    """Broken: autograd.grad w.r.t. the inputs, so fn's parameters get nothing."""

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = [t.detach().requires_grad_(True) for t in ctx.saved_tensors]
        with torch.enable_grad():
            out = ctx.fn(*inputs)
        grads = torch.autograd.grad(out, inputs, grad_outputs, allow_unused=True)
        return (None, *grads)


def ckpt_inputs_only(fn, *args):
    return CkptFnInputsOnly.apply(fn, *args)


# ------------------------------------------------------------- kl-divergence
def kl_categorical(p, q):
    return (torch.xlogy(p, p) - torch.xlogy(p, q)).sum(-1)


def kl_from_logits(pl, ql):
    lp, lq = F.log_softmax(pl, -1), F.log_softmax(ql, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def kl_from_logits_lse(pl, ql):
    """Correct, but normalizes by hand with logsumexp."""
    lp = pl - torch.logsumexp(pl, -1, keepdim=True)
    lq = ql - torch.logsumexp(ql, -1, keepdim=True)
    return torch.sum(torch.softmax(pl, -1) * (lp - lq), dim=-1)


def kl_from_logits_naive(pl, ql):
    """Broken: goes through probabilities, so confident logits underflow to log(0)."""
    p, q = pl.softmax(-1), ql.softmax(-1)
    return (p * torch.log(p / q)).sum(-1)


def my_kl_div(inp, tgt, reduction="mean", log_target=False):
    pw = tgt.exp() * (tgt - inp) if log_target else torch.xlogy(tgt, tgt) - tgt * inp
    return {"none": lambda: pw, "sum": pw.sum, "mean": pw.mean,
            "batchmean": lambda: pw.sum() / inp.size(0)}[reduction]()


def my_kl_div_mean_is_batchmean(inp, tgt, reduction="mean", log_target=False):
    """Broken: 'fixes' mean to be the true KL, which is not what F.kl_div does."""
    return my_kl_div(inp, tgt, "batchmean" if reduction == "mean" else reduction, log_target)


def js_divergence(pl, ql):
    lp, lq = F.log_softmax(pl, -1), F.log_softmax(ql, -1)
    lm = torch.logsumexp(torch.stack([lp, lq]), 0) - math.log(2)
    return 0.5 * (lp.exp() * (lp - lm)).sum(-1) + 0.5 * (lq.exp() * (lq - lm)).sum(-1)


def fit_gaussian(target_logp, grid, direction, steps=500, lr=0.05, mu_init=1.0, sigma_init=0.5):
    mu = torch.tensor(float(mu_init), requires_grad=True)
    ls = torch.tensor(math.log(sigma_init), requires_grad=True)
    opt = torch.optim.Adam([mu, ls], lr=lr)
    for _ in range(steps):
        ql = -0.5 * ((grid - mu) / ls.exp()) ** 2
        loss = kl_from_logits(target_logp, ql) if direction == "forward" else kl_from_logits(ql, target_logp)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mu.item(), ls.exp().item()


def kl_gaussian(mp, lp, mq, lq):
    return 0.5 * (lq - lp + (lp.exp() + (mp - mq) ** 2) / lq.exp() - 1).sum(-1)


def masked_token_kl(pl, ql, mask):
    m = mask.float()
    return (kl_from_logits(pl, ql) * m).sum() / m.sum().clamp(min=1)


def kl_estimators(logp, logq):
    lr = logq - logp
    return -lr, 0.5 * lr ** 2, lr.exp() - 1 - lr


def kl_estimator_grad(logits, ref, est):
    th = logits.detach().requires_grad_(True)
    lp, lq = F.log_softmax(th, -1), F.log_softmax(ref, -1).detach()
    k = kl_estimators(lp, lq)[int(est[1]) - 1]
    return torch.autograd.grad((lp.exp().detach() * k).sum(), th)[0]


def kl_estimator_grad_no_detach(logits, ref, est):
    """Broken: differentiates through pi(x) as well, i.e. the exact KL, not the estimator-as-loss."""
    th = logits.detach().requires_grad_(True)
    lp, lq = F.log_softmax(th, -1), F.log_softmax(ref, -1)
    k = kl_estimators(lp, lq)[int(est[1]) - 1]
    return torch.autograd.grad((lp.exp() * k).sum(), th)[0]


KL = dict(kl_categorical=kl_categorical, kl_from_logits=kl_from_logits, my_kl_div=my_kl_div,
          js_divergence=js_divergence, fit_gaussian=fit_gaussian, kl_gaussian=kl_gaussian,
          masked_token_kl=masked_token_kl, kl_estimators=kl_estimators,
          kl_estimator_grad=kl_estimator_grad)


# ---------------------------------------------------------------- sft-smollm
def build_sft_example(prompt, response, eos, max_len=None):
    prompt, response = list(map(int, prompt)), list(map(int, response))
    ids, lab = prompt + response + [eos], [-100] * len(prompt) + response + [eos]
    return {"input_ids": torch.tensor(ids[:max_len]), "labels": torch.tensor(lab[:max_len])}


def sft_collate(exs, pad_id):
    """Correct, but built with pad_sequence instead of a preallocated tensor."""
    ids = nn.utils.rnn.pad_sequence([e["input_ids"] for e in exs], batch_first=True, padding_value=pad_id)
    lab = nn.utils.rnn.pad_sequence([e["labels"] for e in exs], batch_first=True, padding_value=-100)
    am = nn.utils.rnn.pad_sequence([torch.ones_like(e["input_ids"]) for e in exs], batch_first=True)
    return {"input_ids": ids, "attention_mask": am, "labels": lab}


def sft_collate_pads_labels_with_pad(exs, pad_id):
    """Broken: labels = input_ids with pad positions masked, so a real eos == pad is lost."""
    b = sft_collate(exs, pad_id)
    b["labels"] = b["labels"].masked_fill(b["input_ids"] == pad_id, -100)
    return b


def sft_loss(logits, labels, num_items_in_batch=None):
    lg, tg = logits[:, :-1], labels[:, 1:]
    s = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), tg.reshape(-1), ignore_index=-100, reduction="sum")
    n = num_items_in_batch if num_items_in_batch is not None else int((tg != -100).sum())
    return s / max(int(n), 1)


def sft_loss_unshifted(logits, labels, num_items_in_batch=None):
    """Broken: scores position t against its own token instead of the next one."""
    s = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum")
    n = num_items_in_batch if num_items_in_batch is not None else int((labels != -100).sum())
    return s / max(int(n), 1)


def sft_train_step(model, mbs, opt, max_grad_norm=None):
    model.train()
    n = sum(int((mb["labels"][:, 1:] != -100).sum()) for mb in mbs)
    opt.zero_grad()
    total = 0.0
    for mb in mbs:
        loss = sft_loss(model(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"]).logits,
                        mb["labels"], n)
        loss.backward()
        total += loss.item()
    if max_grad_norm:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    opt.step()
    return total


def sft_train_step_mean_of_means(model, mbs, opt, max_grad_norm=None):
    """Broken: the classic gradient-accumulation bug — each micro-batch is its own mean."""
    model.train()
    opt.zero_grad()
    total = 0.0
    for mb in mbs:
        loss = sft_loss(model(input_ids=mb["input_ids"]).logits, mb["labels"]) / len(mbs)
        loss.backward()
        total += loss.item()
    opt.step()
    return total


SFT = dict(build_sft_example=build_sft_example, sft_collate=sft_collate, sft_loss=sft_loss,
           sft_train_step=sft_train_step)


CASES = [
    # (label, expected, problem_id, args, kwargs)
    ("kv-cache canonical", True, "kv-cache", (), dict(KVCache=KVCache, CachedAttention=Attn)),
    ("kv-cache divergent", True, "kv-cache", (), dict(KVCache=ListKVCache, CachedAttention=SdpaAttn)),
    ("kv-cache broken", False, "kv-cache", (), dict(KVCache=KVCache, CachedAttention=NoPastAttn)),
    ("softmax correct", True, "triton-fused-softmax", (online_softmax,), {}),
    ("softmax unstable", False, "triton-fused-softmax", (unstable_softmax,), {}),
    ("cnn canonical", True, "cnn", (Cnn,), {}),
    ("cnn divergent", True, "cnn", (DivergentCnn,), {}),
    ("cnn wrong classes", False, "cnn", (WrongClassCount,), {}),
    ("rope correct", True, "rotary-positional-embedding", (rotate_half, rope_apply), {}),
    ("rope no-op", False, "rotary-positional-embedding", (rotate_half, rope_noop), {}),
    ("rope query only", False, "rotary-positional-embedding", (rotate_half, rope_query_only), {}),
    ("ckpt correct", True, "gradient-checkpointing", (CkptFn, ckpt), {}),
    ("ckpt no dummy input", False, "gradient-checkpointing", (CkptFn, ckpt_no_dummy), {}),
    ("ckpt inputs-only grads", False, "gradient-checkpointing", (CkptFnInputsOnly, ckpt_inputs_only), {}),
    ("kl canonical", True, "kl-divergence", (), KL),
    ("kl divergent logsumexp", True, "kl-divergence", (), {**KL, "kl_from_logits": kl_from_logits_lse}),
    ("kl naive softmax", False, "kl-divergence", (), {**KL, "kl_from_logits": kl_from_logits_naive}),
    ("kl 'mean' as batchmean", False, "kl-divergence", (), {**KL, "my_kl_div": my_kl_div_mean_is_batchmean}),
    ("kl grad without detach", False, "kl-divergence", (), {**KL, "kl_estimator_grad": kl_estimator_grad_no_detach}),
    ("sft divergent collate", True, "sft-smollm", (), SFT),
    ("sft labels padded via pad id", False, "sft-smollm", (), {**SFT, "sft_collate": sft_collate_pads_labels_with_pad}),
    ("sft unshifted loss", False, "sft-smollm", (), {**SFT, "sft_loss": sft_loss_unshifted}),
    ("sft mean of micro-batch means", False, "sft-smollm", (), {**SFT, "sft_train_step": sft_train_step_mean_of_means}),
]


def main() -> int:
    bad = []
    for label, expected, pid, args, kwargs in CASES:
        got = check(pid, *args, **kwargs)
        if got != expected:
            bad.append(f"{label}: expected {expected}, got {got}")
    print("=" * 60)
    for b in bad:
        print("MISMATCH", b)
    print(f"{len(CASES) - len(bad)}/{len(CASES)} grader cases behaved as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
