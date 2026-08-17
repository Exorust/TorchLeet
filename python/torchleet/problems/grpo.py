"""grpo — Group Relative Policy Optimization: group sampling, group-relative
advantages, and the clipped + KL-penalised objective.

Nothing here compares against a reference GRPO. The generator is driven with
stub policies whose distributions are known exactly (uniform, or peaked on a
token derived from the prompt), so the sampled tokens and their log-probs are
predictable; the advantage normaliser is probed with rewards whose group
structure makes group-local vs global normalisation distinguishable; and the
loss is probed through its defining behaviours (beta scales the KL term,
positive advantage lowers the loss, the ratio clip flattens the objective).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["generate_group_completions", "compute_group_advantages", "grpo_loss"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Generation: repeat_interleave each prompt group_size times so one prompt's "
    "completions sit in a contiguous block, then sample token by token.",
    "Advantages are normalised INSIDE each group: reshape rewards to "
    "(num_prompts, group_size), subtract the row mean, divide by the row std + eps. "
    "No value model, no global mean.",
    "grpo_loss = -mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)) + beta*KL, "
    "with ratio = exp(policy_logps - old_logps) and A broadcast over the tokens "
    "of its sequence.",
]

VOCAB, SEQ, GROUP, NPROMPT = 10, 6, 4, 3
N = NPROMPT * GROUP


# --------------------------------------------------------------------------
# Stub policies. Same call signature as the notebook's PolicyModel:
#     forward(input_ids, prompt=None, hidden=None) -> (logits, hidden)
# so a solver that threads the hidden state, or re-feeds the whole prefix
# every step, both work.
# --------------------------------------------------------------------------
class _UniformPolicy(nn.Module):
    """Every token equally likely -> every log-prob is exactly -log(VOCAB)."""

    def forward(self, input_ids, prompt=None, hidden=None):
        b, s = input_ids.shape
        return torch.zeros(b, s, VOCAB), torch.zeros(1, b, 1)


class _PromptPolicy(nn.Module):
    """Deterministic: always emits token (prompt % VOCAB), so the generated
    tokens reveal which prompt each row was conditioned on."""

    def __init__(self):
        super().__init__()
        self._prompt = None

    def forward(self, input_ids, prompt=None, hidden=None):
        b, s = input_ids.shape
        if prompt is not None:
            assert prompt.numel() == b, (
                f"the policy was called with {b} rows but {prompt.numel()} prompts — "
                f"expand the prompts to num_prompts * group_size before generating")
            self._prompt = prompt.reshape(b).clone()
        assert self._prompt is not None, \
            "the policy was never given `prompt`, so completions cannot be conditioned"
        tok = self._prompt.long() % VOCAB
        onehot = F.one_hot(tok, VOCAB).float()          # (b, VOCAB)
        logits = onehot.unsqueeze(1).expand(b, s, VOCAB) * 40.0 - 20.0
        return logits, torch.zeros(1, b, 1)


def _gen(ns, policy, prompts):
    out = ns.generate_group_completions(policy, prompts, GROUP, SEQ, VOCAB)
    assert isinstance(out, (tuple, list)) and len(out) == 3, (
        "generate_group_completions must return "
        "(sequences, log_probs, prompt_targets)")
    return out


# --------------------------------------------------------------------------
# generate_group_completions
# --------------------------------------------------------------------------
def check_generation_shapes(ns):
    prompts = torch.arange(1.0, NPROMPT + 1)
    seqs, logps, targets = _gen(ns, _UniformPolicy(), prompts)
    assert tuple(seqs.shape) == (N, SEQ), (
        f"sequences should be (num_prompts*group_size, seq_len) = {(N, SEQ)}, "
        f"got {tuple(seqs.shape)}")
    assert tuple(logps.shape) == (N, SEQ), (
        f"log_probs should be one per generated token {(N, SEQ)}, "
        f"got {tuple(logps.shape)}")
    assert tuple(targets.shape) == (N,), (
        f"prompt_targets should be the repeated prompts {(N,)}, "
        f"got {tuple(targets.shape)}")
    assert seqs.dtype in (torch.long, torch.int64, torch.int32), \
        f"sequences must be token ids (integer dtype), got {seqs.dtype}"
    assert int(seqs.min()) >= 0 and int(seqs.max()) < VOCAB, (
        f"tokens must lie in [0, {VOCAB}), got "
        f"[{int(seqs.min())}, {int(seqs.max())}]")


def check_log_probs_match_the_sampling_distribution(ns):
    """Under a uniform policy every sampled token has probability 1/VOCAB."""
    seqs, logps, _ = _gen(ns, _UniformPolicy(), torch.arange(1.0, NPROMPT + 1))
    expected = math.log(1.0 / VOCAB)
    got = float(logps.mean())
    assert torch.allclose(logps, torch.full_like(logps, expected), atol=1e-4), (
        f"with uniform logits every token's log-prob must be log(1/{VOCAB}) = "
        f"{expected:.4f}; got values averaging {got:.4f}. The returned log_probs "
        f"are not the log-probs of the tokens that were actually sampled.")


def check_completions_are_sampled_not_greedy(ns):
    """A group of identical greedy completions carries no learning signal."""
    prompts = torch.arange(1.0, NPROMPT + 1)
    torch.manual_seed(0)
    a, _, _ = _gen(ns, _UniformPolicy(), prompts)
    b, _, _ = _gen(ns, _UniformPolicy(), prompts)
    assert not torch.equal(a, b), (
        "two generations from the same uniform policy produced identical tokens — "
        "GRPO needs sampled completions, not argmax/greedy decoding")
    first = a[0].unsqueeze(0).expand_as(a)
    assert not bool((a == first).all()), (
        "every completion in the batch is identical — the group must be sampled "
        "independently or all advantages collapse to zero")


def check_each_prompt_conditions_its_own_group(ns):
    """Prompt i's completions must be rows [i*G, (i+1)*G) and must be
    conditioned on prompt i."""
    prompts = torch.tensor([1.0, 2.0, 3.0])
    seqs, _, targets = _gen(ns, _PromptPolicy(), prompts)

    grouped_t = targets.reshape(NPROMPT, GROUP).float()
    assert torch.allclose(grouped_t, prompts.unsqueeze(1).expand(NPROMPT, GROUP)), (
        "prompt_targets must repeat each prompt group_size times in a contiguous "
        f"block (prompt i in rows [i*{GROUP}, (i+1)*{GROUP})) so "
        f"compute_group_advantages can reshape to (num_prompts, group_size); got "
        f"{targets.tolist()} for prompts {prompts.tolist()}")

    # The stub policy always emits token (prompt % VOCAB).
    expected = (prompts.long() % VOCAB).reshape(NPROMPT, 1, 1).expand(NPROMPT, GROUP, SEQ)
    assert torch.equal(seqs.reshape(NPROMPT, GROUP, SEQ), expected), (
        "the tokens generated for each group do not match the prompt that group "
        "was supposed to be conditioned on — check that the expanded prompts are "
        f"passed to the policy. expected rows of {expected[:, 0, 0].tolist()}, got "
        f"{seqs.reshape(NPROMPT, GROUP, SEQ)[:, 0, 0].tolist()}")


# --------------------------------------------------------------------------
# compute_group_advantages
# --------------------------------------------------------------------------
def check_advantages_are_normalised_within_each_group(ns):
    torch.manual_seed(1)
    rewards = torch.randn(N) * 7.0 + 3.0
    adv = ns.compute_group_advantages(rewards, GROUP)
    assert tuple(adv.shape) == (N,), \
        f"advantages should be one per completion {(N,)}, got {tuple(adv.shape)}"
    g = adv.reshape(NPROMPT, GROUP)
    means = g.mean(dim=1)
    assert torch.allclose(means, torch.zeros(NPROMPT), atol=1e-4), (
        f"each group's advantages must have mean 0 after subtracting the group "
        f"mean; got per-group means {means.tolist()}")
    stds = g.std(dim=1, unbiased=False)
    assert bool(((stds > 0.7) & (stds < 1.4)).all()), (
        f"each group's advantages must be divided by that group's std (so the "
        f"spread is ~1); got per-group std {stds.tolist()}")


def check_normalisation_is_per_group_not_global(ns):
    """Two groups with the same shape of rewards but wildly different scale must
    come out with the same advantages."""
    small = torch.tensor([0.0, 1.0, 2.0, 3.0])
    big = small * 100.0
    rewards = torch.cat([small, big, small])
    adv = ns.compute_group_advantages(rewards, 4)
    a, b = adv[:4], adv[4:8]
    assert torch.allclose(a, b, atol=1e-3), (
        "group 1's rewards are exactly 100x group 0's, so after per-group "
        f"normalisation both groups must get the same advantages; got {a.tolist()} "
        f"vs {b.tolist()}. This looks like a single global mean/std instead of "
        "one per group.")


def check_constant_rewards_do_not_blow_up(ns):
    adv = ns.compute_group_advantages(torch.full((N,), 5.0), GROUP)
    assert torch.isfinite(adv).all(), (
        "all rewards in a group were equal, so the group std is 0 — add an eps to "
        f"the denominator; got {adv.tolist()}")
    assert torch.allclose(adv, torch.zeros(N), atol=1e-3), (
        f"identical rewards carry no preference signal, so every advantage should "
        f"be ~0; got {adv.tolist()}")


def check_advantages_preserve_reward_order(ns):
    torch.manual_seed(2)
    rewards = torch.randn(N)
    adv = ns.compute_group_advantages(rewards, GROUP)
    r = rewards.reshape(NPROMPT, GROUP)
    a = adv.reshape(NPROMPT, GROUP)
    assert torch.equal(r.argsort(dim=1), a.argsort(dim=1)), (
        "normalisation must be a positive affine map, so the best-rewarded "
        "completion in a group must also have the largest advantage; ranking "
        f"changed (rewards {r[0].tolist()} -> advantages {a[0].tolist()})")


# --------------------------------------------------------------------------
# grpo_loss
# --------------------------------------------------------------------------
B_, S_ = 4, 3


def _loss(ns, policy, old, adv, ref, clip=0.2, beta=0.0):
    out = ns.grpo_loss(policy_logps=policy, old_logps=old, advantages=adv,
                       clip_epsilon=clip, beta=beta, ref_logps=ref)
    assert isinstance(out, dict), \
        f"grpo_loss should return a dict of loss terms, got {type(out).__name__}"
    for key in ("policy_loss", "kl_div", "total_loss"):
        assert key in out, \
            f"grpo_loss's dict is missing '{key}' (needs policy_loss, kl_div, total_loss)"
    return out


def check_loss_returns_scalar_terms(ns):
    torch.manual_seed(3)
    p = torch.randn(B_, S_)
    out = _loss(ns, p, p.clone(), torch.randn(B_), p.clone())
    for key in ("policy_loss", "kl_div", "total_loss"):
        t = torch.as_tensor(out[key])
        assert t.ndim == 0, \
            f"'{key}' should be a scalar, got shape {tuple(t.shape)}"
        assert torch.isfinite(t), f"'{key}' is not finite ({float(t)})"


def check_kl_term_measures_divergence_from_the_reference(ns):
    torch.manual_seed(4)
    p = torch.randn(B_, S_)
    adv = torch.randn(B_)
    same = float(torch.as_tensor(_loss(ns, p, p, adv, p)["kl_div"]))
    assert abs(same) < 1e-5, (
        f"policy_logps == ref_logps means zero divergence, so kl_div should be 0; "
        f"got {same:.5f}")
    near = float(torch.as_tensor(_loss(ns, p + 0.2, p, adv, p)["kl_div"]))
    far = float(torch.as_tensor(_loss(ns, p + 1.0, p, adv, p)["kl_div"]))
    assert far > near > 0, (
        "kl_div must grow as the policy moves away from the reference; got "
        f"{near:.5f} at a gap of 0.2 and {far:.5f} at a gap of 1.0")


def check_beta_scales_the_kl_penalty(ns):
    torch.manual_seed(5)
    p, ref = torch.randn(B_, S_), torch.randn(B_, S_)
    old, adv = torch.randn(B_, S_), torch.randn(B_)
    base = _loss(ns, p, old, adv, ref, beta=0.0)
    pen = _loss(ns, p, old, adv, ref, beta=0.5)
    kl = float(torch.as_tensor(base["kl_div"]))
    delta = float(torch.as_tensor(pen["total_loss"])) - float(torch.as_tensor(base["total_loss"]))
    assert abs(delta - 0.5 * kl) < 1e-4, (
        f"total_loss must add beta * kl_div: raising beta from 0 to 0.5 should "
        f"change total_loss by {0.5 * kl:.5f}, it changed by {delta:.5f}")


def check_positive_advantage_lowers_the_loss(ns):
    """The direction the objective exists to push."""
    p = torch.zeros(B_, S_)
    good = float(torch.as_tensor(
        _loss(ns, p, p, torch.ones(B_), p)["total_loss"]))
    bad = float(torch.as_tensor(
        _loss(ns, p, p, -torch.ones(B_), p)["total_loss"]))
    assert good < bad, (
        "with ratio 1 everywhere, completions with advantage +1 must give a LOWER "
        f"total_loss than completions with advantage -1; got {good:.4f} vs "
        f"{bad:.4f} (the objective is being maximised in the wrong direction)")


def check_clipping_caps_the_objective(ns):
    """Past 1+eps the ratio must stop buying improvement — that IS the clip."""
    old = torch.zeros(B_, S_)
    adv = torch.ones(B_)
    at_1 = float(torch.as_tensor(
        _loss(ns, old, old, adv, old, clip=0.2)["total_loss"]))
    at_1_5 = float(torch.as_tensor(
        _loss(ns, torch.full((B_, S_), math.log(1.5)), old, adv, old, clip=0.2)["total_loss"]))
    at_3 = float(torch.as_tensor(
        _loss(ns, torch.full((B_, S_), math.log(3.0)), old, adv, old, clip=0.2)["total_loss"]))
    assert at_1_5 < at_1 - 1e-4, (
        f"with advantage +1, moving the ratio from 1.0 to 1.5 should improve "
        f"(lower) the loss up to the clip; got {at_1:.5f} -> {at_1_5:.5f}")
    assert abs(at_1_5 - at_3) < 1e-4, (
        f"ratios 1.5 and 3.0 are both beyond 1+clip_epsilon=1.2, so min(r*A, "
        f"clip(r)*A) must give the SAME value; got {at_1_5:.5f} vs {at_3:.5f} — "
        f"the clip is missing or applied to the wrong branch")


def check_gradients_reach_the_policy(ns):
    torch.manual_seed(6)
    p = torch.randn(B_, S_, requires_grad=True)
    out = _loss(ns, p, p.detach(), torch.randn(B_), p.detach(), beta=0.05)
    torch.as_tensor(out["total_loss"]).backward()
    assert p.grad is not None and bool((p.grad != 0).any()), \
        "no gradient reached policy_logps — total_loss is detached from the policy"


CHECKS = [
    check_generation_shapes,
    check_log_probs_match_the_sampling_distribution,
    check_completions_are_sampled_not_greedy,
    check_each_prompt_conditions_its_own_group,
    check_advantages_are_normalised_within_each_group,
    check_normalisation_is_per_group_not_global,
    check_constant_rewards_do_not_blow_up,
    check_advantages_preserve_reward_order,
    check_loss_returns_scalar_terms,
    check_kl_term_measures_divergence_from_the_reference,
    check_beta_scales_the_kl_penalty,
    check_positive_advantage_lowers_the_loss,
    check_clipping_caps_the_objective,
    check_gradients_reach_the_policy,
]
