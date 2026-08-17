"""ppo-rlhf — rollout generation, GAE, and the PPO clipped objective with a
value baseline and a KL penalty.

The heavy check is a pipeline self-consistency one: feed generate_sequences'
own output straight into ppo_step with the *same, unmodified* policy. The
importance ratio must then be exactly 1 at every position, which pins the whole
loss to a value we can write down in closed form. Any misalignment between the
log-probs returned at generation time and the ones ppo_step recomputes shows up
immediately. GAE is probed at its limiting cases (gamma=lam=0 is the TD error,
gamma=lam=1 is the Monte-Carlo return minus the baseline), which are identities,
not a second implementation.
"""
import math

import torch
import torch.nn as nn

ENTRIES = ["generate_sequences", "compute_gae", "ppo_step"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "generate_sequences starts from a BOS token (id = vocab_size), samples one "
    "token at a time from Categorical(logits=logits[:, -1, :]), and returns the "
    "running input ids so the value model can be scored on the same prefixes.",
    "GAE runs backwards: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t) with "
    "V(s_T)=0, then A_t = delta_t + gamma*lam*A_{t+1}.",
    "ppo_step recomputes log-probs for `actions` under the current policy, forms "
    "ratio = exp(new - old), and takes min(ratio*A, clamp(ratio, 1-eps, 1+eps)*A). "
    "total_loss = -policy_objective + 0.5*value_loss + kl_beta*kl_div.",
]

VOCAB, SEQ, BATCH = 12, 5, 6


# --------------------------------------------------------------------------
# stub models: policy(input_ids) -> (B, S, VOCAB); value(input_ids) -> (B, S)
# --------------------------------------------------------------------------
class _UniformPolicy(nn.Module):
    """Flat logits, so every sampled token has log-prob exactly -log(VOCAB)."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        b, s = input_ids.shape
        return torch.zeros(b, s, VOCAB) + self.bias


class _TokenPolicy(nn.Module):
    """Logits depend on the token at each position, so a one-off index shift in
    ppo_step changes the recomputed log-probs."""

    def __init__(self, seed=0, scale=1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(VOCAB + 1, VOCAB)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(VOCAB + 1, VOCAB, generator=g) * scale)

    def forward(self, input_ids):
        return self.emb(input_ids)


class _ZeroValue(nn.Module):
    """Predicts 0 everywhere (but differentiably), so value_loss = mean(returns^2)."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        b, s = input_ids.shape
        return torch.zeros(b, s) + self.bias


def _gen(ns, policy, batch=BATCH, seq=SEQ):
    out = ns.generate_sequences(policy, batch, seq, VOCAB)
    assert isinstance(out, (tuple, list)) and len(out) == 3, \
        "generate_sequences must return (sequences, log_probs, all_input_ids)"
    return out


# --------------------------------------------------------------------------
# generate_sequences
# --------------------------------------------------------------------------
def check_rollout_shapes_and_token_range(ns):
    seqs, logps, ids = _gen(ns, _UniformPolicy())
    assert tuple(seqs.shape) == (BATCH, SEQ), \
        f"sequences should be {(BATCH, SEQ)}, got {tuple(seqs.shape)}"
    assert tuple(logps.shape) == (BATCH, SEQ), \
        f"log_probs should be one per generated token {(BATCH, SEQ)}, got {tuple(logps.shape)}"
    assert seqs.dtype in (torch.long, torch.int64, torch.int32), \
        f"sequences must be token ids (integer dtype), got {seqs.dtype}"
    assert int(seqs.min()) >= 0 and int(seqs.max()) < VOCAB, (
        f"generated tokens must be in [0, {VOCAB}) — the BOS id {VOCAB} is an input "
        f"only; got range [{int(seqs.min())}, {int(seqs.max())}]")
    assert ids.dim() == 2 and ids.shape[0] == BATCH and ids.shape[1] in (SEQ, SEQ + 1), (
        f"all_input_ids should be the BOS-prefixed prefix the models are scored on, "
        f"i.e. {(BATCH, SEQ + 1)} (or {(BATCH, SEQ)}); got {tuple(ids.shape)}")


def check_rollout_inputs_are_bos_then_the_generated_tokens(ns):
    seqs, _, ids = _gen(ns, _UniformPolicy())
    assert bool((ids[:, 0] == VOCAB).all()), (
        f"all_input_ids must start from the BOS token (id = vocab_size = {VOCAB}); "
        f"got first column {ids[:, 0].tolist()}")
    k = ids.shape[1] - 1
    assert torch.equal(ids[:, 1:], seqs[:, :k]), (
        "after BOS, all_input_ids must be the tokens that were actually generated "
        "(shifted by one) — otherwise the value model and ppo_step score a "
        "different sequence than the one that was sampled")


def check_rollout_log_probs_match_the_policy(ns):
    _, logps, _ = _gen(ns, _UniformPolicy())
    expected = math.log(1.0 / VOCAB)
    assert torch.allclose(logps, torch.full_like(logps, expected), atol=1e-4), (
        f"under a uniform policy every token's log-prob is log(1/{VOCAB}) = "
        f"{expected:.4f}; got values averaging {float(logps.mean()):.4f}")


def check_rollout_is_sampled_not_greedy(ns):
    p = _UniformPolicy()
    torch.manual_seed(0)
    a, _, _ = _gen(ns, p)
    b, _, _ = _gen(ns, p)
    assert not torch.equal(a, b), (
        "two rollouts from the same uniform policy came out identical — PPO needs "
        "sampled actions (torch.distributions.Categorical(...).sample()), not argmax")


# --------------------------------------------------------------------------
# compute_gae
# --------------------------------------------------------------------------
def _rv(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(BATCH, SEQ, generator=g),
            torch.randn(BATCH, SEQ, generator=g))


def check_gae_shape(ns):
    r, v = _rv()
    adv = ns.compute_gae(r, v, gamma=0.99, lam=0.95)
    assert tuple(adv.shape) == (BATCH, SEQ), \
        f"advantages should be per-step {(BATCH, SEQ)}, got {tuple(adv.shape)}"
    assert torch.isfinite(adv).all(), "advantages contain NaN/Inf"


def check_gae_with_no_discount_is_the_td_error(ns):
    """gamma=0, lam=0 collapses GAE to r_t - V(s_t)."""
    r, v = _rv(1)
    adv = ns.compute_gae(r, v, gamma=0.0, lam=0.0)
    assert torch.allclose(adv, r - v, atol=1e-5), (
        "with gamma=0 and lam=0 every advantage is just r_t - V(s_t); "
        f"max deviation {float((adv - (r - v)).abs().max()):.5f}")


def check_gae_lambda_zero_is_one_step_bootstrap(ns):
    """lam=0 keeps only delta_t = r_t + gamma*V(s_{t+1}) - V(s_t), V(s_T)=0."""
    r, v = _rv(2)
    gamma = 0.9
    nxt = torch.cat([v[:, 1:], torch.zeros(BATCH, 1)], dim=1)
    expected = r + gamma * nxt - v
    adv = ns.compute_gae(r, v, gamma=gamma, lam=0.0)
    assert torch.allclose(adv, expected, atol=1e-5), (
        "with lam=0 the advantage is the one-step TD residual "
        "r_t + gamma*V(s_{t+1}) - V(s_t) (with V past the last step = 0); "
        f"max deviation {float((adv - expected).abs().max()):.5f}")


def check_gae_lambda_one_is_return_minus_baseline(ns):
    """gamma=lam=1 telescopes to (sum of future rewards) - V(s_t)."""
    r, v = _rv(3)
    future = r.flip(1).cumsum(1).flip(1)
    expected = future - v
    adv = ns.compute_gae(r, v, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, expected, atol=1e-4), (
        "with gamma=lam=1 the GAE sum telescopes to the Monte-Carlo return minus "
        "the baseline, sum_{k>=t} r_k - V(s_t); max deviation "
        f"{float((adv - expected).abs().max()):.5f}. Check that the recursion runs "
        "backwards and that V past the final step is 0.")


def check_gae_is_a_backward_recursion(ns):
    """Later rewards must reach earlier advantages, never the other way round."""
    r = torch.zeros(1, SEQ)
    r[0, -1] = 1.0                       # reward only at the terminal step
    v = torch.zeros(1, SEQ)
    adv = ns.compute_gae(r, v, gamma=0.9, lam=0.9)
    assert bool((adv > 0).all()), (
        "a single reward at the last step must propagate back to every earlier "
        f"advantage through the backward recursion; got {adv[0].tolist()}")
    diffs = adv[0, 1:] - adv[0, :-1]
    assert bool((diffs > 0).all()), (
        "with gamma*lam < 1 the credit for a terminal reward must decay the further "
        f"back you go, so advantages should increase with t; got {adv[0].tolist()}")


# --------------------------------------------------------------------------
# ppo_step
# --------------------------------------------------------------------------
KEYS = ("policy_loss", "value_loss", "kl_div", "total_loss")


def _step(ns, policy, value, ref, old, ids, actions, adv, returns,
          clip=0.2, kl_beta=0.01):
    out = ns.ppo_step(policy, value, ref, old, ids, actions, adv, returns,
                      clip_epsilon=clip, kl_beta=kl_beta)
    assert isinstance(out, dict), \
        f"ppo_step should return a dict of loss terms, got {type(out).__name__}"
    for k in KEYS:
        assert k in out, f"ppo_step's dict is missing '{k}' (needs {', '.join(KEYS)})"
    return {k: torch.as_tensor(v) for k, v in out.items()}


def _rollout(ns, policy):
    with torch.no_grad():
        seqs, old, ids = _gen(ns, policy)
    return seqs, old, ids


def check_ppo_step_returns_scalar_terms(ns):
    p = _TokenPolicy()
    seqs, old, ids = _rollout(ns, p)
    adv = torch.randn(BATCH, SEQ)
    out = _step(ns, p, _ZeroValue(), _TokenPolicy(), old, ids, seqs, adv,
                torch.zeros(BATCH, SEQ))
    for k in KEYS:
        assert out[k].ndim == 0, f"'{k}' should be a scalar, got shape {tuple(out[k].shape)}"
        assert torch.isfinite(out[k]), f"'{k}' is not finite ({float(out[k])})"
    assert float(out["value_loss"]) >= 0, \
        f"value_loss is an MSE and cannot be negative, got {float(out['value_loss']):.4f}"


def check_unchanged_policy_gives_ratio_one(ns):
    """The pipeline check: generate, then score the SAME policy. Nothing moved,
    so the ratio is 1 and total_loss is exactly -mean(advantages)."""
    p = _TokenPolicy(seed=5)
    seqs, old, ids = _rollout(ns, p)
    adv = torch.randn(BATCH, SEQ)
    out = _step(ns, p, _ZeroValue(), p, old, ids, seqs, adv,
                torch.zeros(BATCH, SEQ), clip=0.2, kl_beta=0.01)
    assert abs(float(out["kl_div"])) < 1e-5, (
        "ref_policy IS the policy here, so kl_div must be 0; got "
        f"{float(out['kl_div']):.5f} — the reference log-probs are not being "
        "computed on the same actions/positions")
    assert abs(float(out["value_loss"])) < 1e-5, (
        "the value model predicts 0 and the returns are 0, so value_loss must be 0; "
        f"got {float(out['value_loss']):.5f}")
    expected = -float(adv.mean())
    got = float(out["total_loss"])
    assert abs(got - expected) < 1e-4, (
        f"the policy has not been updated since generation, so exp(new - old) = 1 "
        f"at every position and the clipped objective is just mean(advantages). "
        f"total_loss should be {expected:.5f}, got {got:.5f}. Usually this means "
        f"the log-probs ppo_step recomputes are off by one position from the ones "
        f"generate_sequences returned.")


def check_value_loss_is_mse_against_returns(ns):
    p = _TokenPolicy()
    seqs, old, ids = _rollout(ns, p)
    returns = torch.full((BATCH, SEQ), 3.0)
    out = _step(ns, p, _ZeroValue(), p, old, ids, seqs,
                torch.zeros(BATCH, SEQ), returns)
    assert abs(float(out["value_loss"]) - 9.0) < 1e-3, (
        "the value model predicts 0 and the returns are all 3, so the MSE value "
        f"loss must be 9.0; got {float(out['value_loss']):.4f}")


def check_kl_penalty_is_scaled_by_kl_beta(ns):
    p = _TokenPolicy(seed=1)
    ref = _TokenPolicy(seed=2)
    seqs, old, ids = _rollout(ns, p)
    adv, returns = torch.randn(BATCH, SEQ), torch.zeros(BATCH, SEQ)
    a = _step(ns, p, _ZeroValue(), ref, old, ids, seqs, adv, returns, kl_beta=0.0)
    b = _step(ns, p, _ZeroValue(), ref, old, ids, seqs, adv, returns, kl_beta=1.0)
    kl = float(a["kl_div"])
    assert abs(kl) > 1e-4, (
        "policy and ref_policy have different weights, so kl_div should be nonzero; "
        f"got {kl:.6f}")
    delta = float(b["total_loss"]) - float(a["total_loss"])
    assert abs(delta - kl) < 1e-4, (
        f"total_loss must add kl_beta * kl_div: raising kl_beta from 0 to 1 should "
        f"move total_loss by {kl:.5f}, it moved by {delta:.5f}")


def check_positive_advantage_lowers_the_loss(ns):
    p = _TokenPolicy(seed=3)
    seqs, old, ids = _rollout(ns, p)
    returns = torch.zeros(BATCH, SEQ)
    good = float(_step(ns, p, _ZeroValue(), p, old, ids, seqs,
                       torch.ones(BATCH, SEQ), returns)["total_loss"])
    bad = float(_step(ns, p, _ZeroValue(), p, old, ids, seqs,
                      -torch.ones(BATCH, SEQ), returns)["total_loss"])
    assert good < bad, (
        "actions with advantage +1 must give a LOWER total_loss than actions with "
        f"advantage -1; got {good:.4f} vs {bad:.4f} (the surrogate objective is "
        "being optimised in the wrong direction)")


def check_clipping_caps_the_objective(ns):
    """Shrinking old_log_probs pushes the ratio up; past 1+eps it must stop paying."""
    p = _TokenPolicy(seed=4)
    seqs, old, ids = _rollout(ns, p)
    adv, returns = torch.ones(BATCH, SEQ), torch.zeros(BATCH, SEQ)

    def total(ratio):
        return float(_step(ns, p, _ZeroValue(), p, old - math.log(ratio), ids, seqs,
                           adv, returns, clip=0.2, kl_beta=0.0)["total_loss"])

    at_1, at_15, at_3 = total(1.0), total(1.5), total(3.0)
    assert at_15 < at_1 - 1e-4, (
        f"with advantage +1, raising the ratio from 1.0 to 1.5 should lower the "
        f"loss up to the clip; got {at_1:.5f} -> {at_15:.5f}")
    assert abs(at_15 - at_3) < 1e-4, (
        f"ratios 1.5 and 3.0 are both past 1+clip_epsilon=1.2, so "
        f"min(r*A, clamp(r)*A) must return the same value; got {at_15:.5f} vs "
        f"{at_3:.5f} — the clip is missing or on the wrong branch")


def check_gradients_reach_policy_and_value_models(ns):
    p = _TokenPolicy(seed=7)
    v = _ZeroValue()
    seqs, old, ids = _rollout(ns, p)
    out = _step(ns, p, v, _TokenPolicy(seed=8), old, ids, seqs,
                torch.randn(BATCH, SEQ), torch.randn(BATCH, SEQ), kl_beta=0.01)
    out["total_loss"].backward()
    assert p.emb.weight.grad is not None and bool((p.emb.weight.grad != 0).any()), \
        "no gradient reached the policy — the surrogate objective is detached"
    assert v.bias.grad is not None and bool((v.bias.grad != 0).any()), \
        "no gradient reached the value model — value_loss is missing from total_loss"


CHECKS = [
    check_rollout_shapes_and_token_range,
    check_rollout_inputs_are_bos_then_the_generated_tokens,
    check_rollout_log_probs_match_the_policy,
    check_rollout_is_sampled_not_greedy,
    check_gae_shape,
    check_gae_with_no_discount_is_the_td_error,
    check_gae_lambda_zero_is_one_step_bootstrap,
    check_gae_lambda_one_is_return_minus_baseline,
    check_gae_is_a_backward_recursion,
    check_ppo_step_returns_scalar_terms,
    check_unchanged_policy_gives_ratio_one,
    check_value_loss_is_mse_against_returns,
    check_kl_penalty_is_scaled_by_kl_beta,
    check_positive_advantage_lowers_the_loss,
    check_clipping_caps_the_objective,
    check_gradients_reach_policy_and_value_models,
]
