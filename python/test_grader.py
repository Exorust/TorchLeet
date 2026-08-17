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
