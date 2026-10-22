"""kl-divergence — every way KL shows up: probabilities, logits, F.kl_div semantics,
JS, forward vs reverse, Gaussians, masked token-level KL, and k1/k2/k3 estimators."""
import math
import warnings

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence

ENTRIES = ["kl_categorical", "kl_from_logits", "my_kl_div", "js_divergence", "fit_gaussian",
           "kl_gaussian", "masked_token_kl", "kl_estimators", "kl_estimator_grad"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Everything is sum(p * (log p - log q)) over the last dim. Use torch.xlogy when you have "
    "probabilities and F.log_softmax when you have logits, so 0 * log 0 and softmax underflow "
    "never turn into nan.",
    "F.kl_div(input, target) is KL(target || exp(input)): input is the model's log-probs, the "
    "target comes first in the math, and 'batchmean' (sum / batch size) is the true KL while "
    "'mean' divides by every element.",
    "For samples x ~ p with log r = log q(x) - log p(x): k1 = -log r, k2 = (log r)^2 / 2, "
    "k3 = (r - 1) - log r. For kl_estimator_grad, weight every category by pi.detach() and let "
    "the gradient flow only through log pi.",
]


def _oracle_kl(p_logits, q_logits):
    return kl_divergence(Categorical(logits=p_logits), Categorical(logits=q_logits))


def _t(x):
    return torch.as_tensor(x)


# ------------------------------------------------------------- probabilities
def check_categorical_matches_torch(ns):
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(4, 3, 6), -1)
    q = torch.softmax(torch.randn(4, 3, 6), -1)
    got = _t(ns.kl_categorical(p, q))
    assert tuple(got.shape) == (4, 3), \
        f"kl_categorical should reduce only the last dim: expected (4, 3), got {tuple(got.shape)}"
    ref = kl_divergence(Categorical(probs=p), Categorical(probs=q))
    assert torch.allclose(got, ref, atol=1e-5), \
        "kl_categorical(p, q) disagrees with KL(P || Q); check the direction and the formula"


def check_categorical_handles_zeros(ns):
    p = torch.tensor([0.5, 0.5, 0.0])
    q = torch.tensor([0.25, 0.25, 0.5])
    got = _t(ns.kl_categorical(p, q))
    assert not torch.isnan(got), "p = 0 entries produced nan; 0 * log 0 must count as 0 (try torch.xlogy)"
    assert abs(float(got) - math.log(2)) < 1e-5, f"expected log 2 = {math.log(2):.4f}, got {float(got):.4f}"
    back = _t(ns.kl_categorical(q, p))
    assert torch.isinf(back) and back > 0, \
        f"q = 0 where p > 0 must give +inf, got {float(back)}"


# -------------------------------------------------------------------- logits
def check_logits_match_torch(ns):
    torch.manual_seed(1)
    pl, ql = torch.randn(3, 5, 10), torch.randn(3, 5, 10)
    got = _t(ns.kl_from_logits(pl, ql))
    assert tuple(got.shape) == (3, 5), f"expected shape (3, 5), got {tuple(got.shape)}"
    assert torch.allclose(got, _oracle_kl(pl, ql), atol=1e-5), \
        "kl_from_logits(p_logits, q_logits) disagrees with KL(softmax(p) || softmax(q))"


def check_logits_stable_for_large_logits(ns):
    torch.manual_seed(2)
    pl, ql = 100 * torch.randn(4, 12), 100 * torch.randn(4, 12)
    got = _t(ns.kl_from_logits(pl, ql))
    assert torch.isfinite(got).all(), \
        "kl_from_logits returned inf/nan for large logits; compute log-probs with F.log_softmax"
    lp, lq = F.log_softmax(pl.double(), -1), F.log_softmax(ql.double(), -1)
    ref = (lp.exp() * (lp - lq)).sum(-1)
    assert torch.allclose(got.double(), ref, rtol=1e-3, atol=1e-3), \
        "large-logit KL is finite but wrong; is a probability underflowing somewhere?"


# ------------------------------------------------------------------- F.kl_div
def check_kl_div_matches_torch_semantics(ns):
    torch.manual_seed(3)
    for shape in ((8, 5), (2, 3, 7)):
        log_q = F.log_softmax(torch.randn(*shape), -1)
        p = torch.softmax(torch.randn(*shape), -1)
        p_zero = p.clone()
        p_zero[..., 0] = 0.0
        p_zero = p_zero / p_zero.sum(-1, keepdim=True)
        for log_target, tgt in ((False, p_zero), (True, p.log())):
            for red in ("none", "sum", "mean", "batchmean"):
                got = _t(ns.my_kl_div(log_q, tgt, reduction=red, log_target=log_target))
                with warnings.catch_warnings():   # torch warns on 'mean'; that is the point here
                    warnings.simplefilter("ignore")
                    exp = F.kl_div(log_q, tgt, reduction=red, log_target=log_target)
                assert tuple(got.shape) == tuple(exp.shape) and torch.allclose(got, exp, atol=1e-6), (
                    f"my_kl_div(reduction='{red}', log_target={log_target}) on shape {shape} "
                    "does not match F.kl_div; remember 'mean' averages over every element and "
                    "'batchmean' divides by input.size(0)")


# ------------------------------------------------------------------------ JS
def check_js_divergence(ns):
    torch.manual_seed(4)
    pl, ql = torch.randn(6, 7), torch.randn(6, 7)
    got = _t(ns.js_divergence(pl, ql))
    P, Q = pl.softmax(-1), ql.softmax(-1)
    M = Categorical(probs=0.5 * (P + Q))
    ref = 0.5 * kl_divergence(Categorical(probs=P), M) + 0.5 * kl_divergence(Categorical(probs=Q), M)
    assert tuple(got.shape) == (6,), f"expected shape (6,), got {tuple(got.shape)}"
    assert torch.allclose(got, ref, atol=1e-5), "js_divergence disagrees with 0.5*KL(P||M) + 0.5*KL(Q||M)"
    assert torch.allclose(got, _t(ns.js_divergence(ql, pl)), atol=1e-6), "JS must be symmetric"
    far_p = torch.tensor([50.0, -50.0, -50.0, -50.0])
    far_q = torch.tensor([-50.0, -50.0, -50.0, 50.0])
    far = float(_t(ns.js_divergence(far_p, far_q)))
    assert abs(far - math.log(2)) < 1e-4, f"disjoint supports must give log 2, got {far:.4f}"


# ---------------------------------------------------------- forward vs reverse
def check_forward_covers_reverse_seeks(ns):
    grid = torch.linspace(-6, 6, 241)
    mix = torch.logsumexp(torch.stack([Normal(-2.5, 0.6).log_prob(grid),
                                       Normal(2.5, 0.6).log_prob(grid)]), 0)
    target = F.log_softmax(mix, -1)
    kw = dict(steps=500, lr=0.05, mu_init=1.0, sigma_init=0.5)
    mu_f, s_f = (float(v) for v in ns.fit_gaussian(target, grid, "forward", **kw))
    mu_r, s_r = (float(v) for v in ns.fit_gaussian(target, grid, "reverse", **kw))
    assert abs(mu_f) < 0.5 and s_f > 2.0, (
        f"forward KL(target || q) should cover both modes (mu ~ 0, wide sigma); got mu={mu_f:.2f}, "
        f"sigma={s_f:.2f}")
    assert abs(mu_r - 2.5) < 0.3 and s_r < 1.0, (
        f"reverse KL(q || target) from mu_init=1.0 should lock onto the +2.5 mode with a narrow sigma; "
        f"got mu={mu_r:.2f}, sigma={s_r:.2f}. Is the direction swapped?")


# ------------------------------------------------------------------ Gaussians
def check_gaussian_closed_form(ns):
    torch.manual_seed(5)
    mp, lp, mq, lq = (torch.randn(4, 3) for _ in range(4))
    got = _t(ns.kl_gaussian(mp, lp, mq, lq))
    ref = kl_divergence(Normal(mp, (0.5 * lp).exp()), Normal(mq, (0.5 * lq).exp())).sum(-1)
    assert tuple(got.shape) == (4,), f"sum over the last dim: expected (4,), got {tuple(got.shape)}"
    assert torch.allclose(got, ref, atol=1e-5), \
        "kl_gaussian disagrees with the Normal closed form; arguments are log-variances"
    z = torch.zeros_like(mp)
    vae = -0.5 * (1 + lp - mp ** 2 - lp.exp()).sum(-1)
    assert torch.allclose(_t(ns.kl_gaussian(mp, lp, z, z)), vae, atol=1e-5), \
        "against N(0, I) it should reduce to the VAE term -0.5 * sum(1 + logvar - mu^2 - exp(logvar))"


# ------------------------------------------------------------- token-level KL
def _token_setup():
    torch.manual_seed(6)
    B, T, V = 3, 6, 11
    pl, ql = torch.randn(B, T, V), torch.randn(B, T, V)
    pl[2] *= 4
    mask = torch.zeros(B, T)
    for i, n in enumerate((6, 4, 2)):
        mask[i, :n] = 1
    return pl, ql, mask


def check_masked_token_kl_is_token_mean(ns):
    pl, ql, mask = _token_setup()
    got = _t(ns.masked_token_kl(pl, ql, mask))
    assert got.ndim == 0, f"expected a scalar, got shape {tuple(got.shape)}"
    ref = _oracle_kl(pl, ql)[mask.bool()].mean()
    assert torch.allclose(got, ref, atol=1e-5), \
        f"expected the mean over the {int(mask.sum())} unmasked tokens ({float(ref):.4f}), got {float(got):.4f}"
    as_bool = _t(ns.masked_token_kl(pl, ql, mask.bool()))
    assert torch.allclose(as_bool, ref, atol=1e-5), "a boolean mask must work too"


def check_masked_positions_are_ignored(ns):
    pl, ql, mask = _token_setup()
    base = _t(ns.masked_token_kl(pl, ql, mask))
    pad = mask == 0
    pl2, ql2 = pl.clone(), ql.clone()
    pl2[pad] = 1e3 * torch.randn(int(pad.sum()), pl.shape[-1])
    ql2[pad] = -1e3 * torch.randn(int(pad.sum()), pl.shape[-1])
    assert torch.allclose(_t(ns.masked_token_kl(pl2, ql2, mask)), base, atol=1e-5), \
        "changing the logits at masked positions changed the result"
    qg = ql.clone().requires_grad_(True)
    _t(ns.masked_token_kl(pl, qg, mask)).backward()
    assert qg.grad is not None and qg.grad[pad].abs().max() == 0, "masked positions must get zero gradient"
    assert qg.grad[~pad].abs().max() > 0, "no gradient reached the unmasked positions"


# ------------------------------------------------------------------ estimators
def check_estimator_formulas(ns):
    torch.manual_seed(7)
    logp, logq = -torch.rand(100) * 3, -torch.rand(100) * 3
    k1, k2, k3 = (_t(k) for k in ns.kl_estimators(logp, logq))
    lr = logq - logp
    for name, got, exp in (("k1", k1, -lr), ("k2", k2, 0.5 * lr ** 2), ("k3", k3, lr.exp() - 1 - lr)):
        assert tuple(got.shape) == (100,), f"{name}: expected one value per sample, got {tuple(got.shape)}"
        assert torch.allclose(got, exp, atol=1e-5), \
            f"{name} is wrong; with log r = logq - logp, check the sign and the definition"


def check_estimators_unbiased_on_samples(ns):
    torch.manual_seed(8)
    pl = torch.randn(50)
    ql = pl + torch.randn(50)
    true = float(_oracle_kl(pl, ql))
    x = Categorical(logits=pl).sample((200_000,))
    k1, _, k3 = (_t(k) for k in ns.kl_estimators(F.log_softmax(pl, -1)[x], F.log_softmax(ql, -1)[x]))
    for name, k in (("k1", k1), ("k3", k3)):
        se = float(k.std()) / math.sqrt(k.numel())
        assert abs(float(k.mean()) - true) < 5 * se, \
            f"{name} should be unbiased for KL(p || q) with x ~ p: mean {float(k.mean()):.4f} vs true {true:.4f}"
    assert (k3 >= -1e-6).all(), "k3 = (r - 1) - log r is non-negative for every sample"


def check_estimator_gradients(ns):
    torch.manual_seed(9)
    logits, ref = torch.randn(4, 10), torch.randn(4, 10)
    before = logits.clone()
    th = logits.clone().requires_grad_(True)
    g_rev = torch.autograd.grad(_oracle_kl(th, ref).sum(), th)[0]
    th = logits.clone().requires_grad_(True)
    g_fwd = torch.autograd.grad(_oracle_kl(ref, th).sum(), th)[0]
    got = {e: _t(ns.kl_estimator_grad(logits, ref, e)) for e in ("k1", "k2", "k3")}
    assert torch.equal(logits, before), "kl_estimator_grad must not modify its input logits"
    for e, g in got.items():
        assert tuple(g.shape) == (4, 10), f"{e}: gradient should have the logits' shape, got {tuple(g.shape)}"
    assert torch.allclose(got["k1"], torch.zeros_like(logits), atol=1e-6), \
        "k1 as a loss has zero expected gradient; are you weighting by pi.detach() and differentiating only k?"
    assert torch.allclose(got["k2"], g_rev, atol=1e-5), \
        "k2 as a loss should give the gradient of the reverse KL(pi || ref)"
    assert torch.allclose(got["k3"], g_fwd, atol=1e-5), \
        "k3 as a loss should give the gradient of the forward KL(ref || pi)"


CHECKS = [check_categorical_matches_torch, check_categorical_handles_zeros,
          check_logits_match_torch, check_logits_stable_for_large_logits,
          check_kl_div_matches_torch_semantics, check_js_divergence,
          check_forward_covers_reverse_seeks, check_gaussian_closed_form,
          check_masked_token_kl_is_token_mean, check_masked_positions_are_ignored,
          check_estimator_formulas, check_estimators_unbiased_on_samples,
          check_estimator_gradients]
