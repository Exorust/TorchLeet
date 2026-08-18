"""ddim-cfg — deterministic DDIM sampling with classifier-free guidance.

Two properties carry most of the weight and neither needs a reference sampler:

* eta = 0 means DDIM is *deterministic*. Same seed in, bit-identical samples
  out, and a step called with t_prev == t must return x_t untouched (the DDIM
  update re-noises the predicted x_0 back to the level it came from).
* CFG is an affine blend, so guidance_scale = 1 must reproduce the plain
  conditional prediction exactly, and a stub denoiser whose output is just its
  class label pins eps_uncond + s*(eps_cond - eps_uncond) in closed form.
"""
import torch
import torch.nn as nn

from torchleet.runner import Skip

ENTRIES = ["ConditionalDenoiser", "ddim_sample_step", "guided_predict",
           "ddim_sample_loop"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "The null/unconditional label is the extra embedding row at index "
    "num_classes, so the class embedding table needs num_classes + 1 entries.",
    "CFG: eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond). At "
    "guidance_scale = 1 that is exactly eps_cond.",
    "DDIM (eta=0): x0_pred = (x_t - sqrt(1-abar_t)*eps) / sqrt(abar_t), then "
    "x_prev = sqrt(abar_prev)*x0_pred + sqrt(1-abar_prev)*eps. No randn anywhere "
    "in the step — the only randomness is the initial x_T.",
]

DIM, NUM_CLASSES, TOTAL = 2, 4, 300

_BETAS = torch.linspace(1e-4, 0.02, TOTAL)
ABAR = torch.cumprod(1.0 - _BETAS, dim=0)


# --------------------------------------------------------------------------
# stub denoisers: forward(x, t, class_label) -> predicted noise
# --------------------------------------------------------------------------
class _LabelDenoiser(nn.Module):
    """Predicts the class label itself, so the CFG blend is exactly computable."""

    def forward(self, x, t, class_label):
        lab = class_label.float().reshape(-1, *([1] * (x.dim() - 1)))
        return lab.expand_as(x).clone()


class _ConstDenoiser(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x, t, class_label):
        return torch.full_like(x, self.c)


class _NonlinearDenoiser(nn.Module):
    """Depends on x, t and the label — used where the check must hold for any model."""

    def forward(self, x, t, class_label):
        lab = class_label.float().reshape(-1, *([1] * (x.dim() - 1)))
        tt = t.float().reshape(-1, *([1] * (x.dim() - 1)))
        return torch.tanh(1.7 * x + 0.3 * lab + 0.01 * tt) + 0.2 * lab


class _RecordingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x, t, class_label):
        self.seen.append(int(t.reshape(-1)[0]))
        return torch.zeros_like(x)


def _numpy_skip(e):
    msg = str(e)
    if "numpy" in msg or "'np'" in msg:
        raise Skip("this notebook's timestep embedding imports numpy — "
                   "pip install numpy to check ConditionalDenoiser")
    raise e


def _cond_denoiser(ns):
    for kw in (dict(data_dim=DIM, num_classes=NUM_CLASSES, hidden_dim=32, time_dim=16),
               dict(data_dim=DIM, num_classes=NUM_CLASSES), dict()):
        try:
            return ns.ConditionalDenoiser(**kw)
        except TypeError:
            continue
        except (NameError, ImportError) as e:
            _numpy_skip(e)
    raise AssertionError("could not build ConditionalDenoiser(data_dim=..., num_classes=...)")


def _labels(b, c=1):
    return torch.full((b,), c, dtype=torch.long)


# --------------------------------------------------------------------------
# ConditionalDenoiser
# --------------------------------------------------------------------------
def check_conditional_denoiser_shape(ns):
    m = _cond_denoiser(ns)
    m.eval()
    x = torch.randn(6, DIM)
    try:
        with torch.no_grad():
            out = m(x, torch.randint(0, TOTAL, (6,)), _labels(6, 2))
    except (NameError, ImportError) as e:
        _numpy_skip(e)
    assert tuple(out.shape) == (6, DIM), (
        f"the denoiser predicts the noise added to x, so the output shape must be "
        f"{(6, DIM)}, got {tuple(out.shape)}")
    assert torch.isfinite(out).all(), "denoiser output contains NaN/Inf"


def check_conditional_denoiser_accepts_the_null_class(ns):
    """CFG needs an unconditional branch: label == num_classes must be legal."""
    m = _cond_denoiser(ns)
    m.eval()
    x = torch.randn(6, DIM)
    t = torch.full((6,), 10)
    try:
        with torch.no_grad():
            uncond = m(x, t, _labels(6, NUM_CLASSES))
            cond = m(x, t, _labels(6, 0))
    except IndexError as e:
        raise AssertionError(
            f"passing the null label (index num_classes = {NUM_CLASSES}) raised "
            f"{type(e).__name__}: {e}. The class embedding table needs "
            f"num_classes + 1 = {NUM_CLASSES + 1} rows so CFG has an "
            f"unconditional branch.")
    except (NameError, ImportError) as e:
        _numpy_skip(e)
    assert not torch.allclose(uncond, cond, atol=1e-6), (
        "the null class and class 0 gave the same prediction — the label is not "
        "reaching the network, so guidance has nothing to amplify")


def check_conditional_denoiser_is_deterministic_in_eval(ns):
    """Label dropout is a *training* trick; sampling must be repeatable."""
    m = _cond_denoiser(ns)
    m.eval()
    x, t, y = torch.randn(64, DIM), torch.full((64,), 5), _labels(64, 3)
    try:
        with torch.no_grad():
            a, b = m(x, t, y), m(x, t, y)
    except (NameError, ImportError) as e:
        _numpy_skip(e)
    assert torch.allclose(a, b, atol=1e-6), (
        "two identical eval-mode calls disagreed — random label dropout must be "
        "guarded by self.training so it does not fire during sampling")


# --------------------------------------------------------------------------
# guided_predict
# --------------------------------------------------------------------------
def check_guided_predict_matches_the_cfg_blend(ns):
    """The stub predicts its own label, so the blend is a number we can write down."""
    m = _LabelDenoiser()
    x = torch.randn(5, DIM)
    t = torch.full((5,), 7)
    for label in (0, 1, 3):
        for s in (0.0, 1.0, 3.0, 7.5):
            got = ns.guided_predict(m, x, t, _labels(5, label), s, NUM_CLASSES)
            expected = NUM_CLASSES + s * (label - NUM_CLASSES)
            assert tuple(got.shape) == (5, DIM), \
                f"guided_predict should keep the noise shape {(5, DIM)}, got {tuple(got.shape)}"
            assert torch.allclose(got, torch.full_like(got, expected), atol=1e-5), (
                f"this denoiser returns its label, so with class {label} and "
                f"guidance_scale {s} the blend eps_uncond + s*(eps_cond - "
                f"eps_uncond) = {NUM_CLASSES} + {s}*({label} - {NUM_CLASSES}) = "
                f"{expected}; got {float(got.flatten()[0]):.4f}. Check that the "
                f"unconditional pass uses label index num_classes = {NUM_CLASSES}.")


def check_guidance_scale_one_is_the_plain_conditional(ns):
    """Holds for any denoiser, not just the stub."""
    m = _NonlinearDenoiser()
    x, t, y = torch.randn(5, DIM), torch.full((5,), 40), _labels(5, 2)
    got = ns.guided_predict(m, x, t, y, 1.0, NUM_CLASSES)
    assert torch.allclose(got, m(x, t, y), atol=1e-6), (
        "at guidance_scale = 1 the blend collapses to eps_cond, so guided_predict "
        "must return exactly the conditional prediction")


def check_guidance_scale_zero_is_unconditional(ns):
    m = _NonlinearDenoiser()
    x, t, y = torch.randn(5, DIM), torch.full((5,), 40), _labels(5, 2)
    got = ns.guided_predict(m, x, t, y, 0.0, NUM_CLASSES)
    null = m(x, t, _labels(5, NUM_CLASSES))
    assert torch.allclose(got, null, atol=1e-6), (
        "at guidance_scale = 0 the blend collapses to eps_uncond, i.e. the "
        "prediction under the null label")


# --------------------------------------------------------------------------
# ddim_sample_step
# --------------------------------------------------------------------------
def _ddim(x, t, t_prev, eps):
    x0 = (x - torch.sqrt(1.0 - ABAR[t]) * eps) / torch.sqrt(ABAR[t])
    if t_prev < 0:
        return x0
    return torch.sqrt(ABAR[t_prev]) * x0 + torch.sqrt(1.0 - ABAR[t_prev]) * eps


def check_ddim_step_is_deterministic(ns):
    m = _NonlinearDenoiser()
    x, y = torch.randn(200, DIM), _labels(200, 2)
    a = ns.ddim_sample_step(m, x, 120, 90, ABAR, y, 1.0, NUM_CLASSES)
    b = ns.ddim_sample_step(m, x, 120, 90, ABAR, y, 1.0, NUM_CLASSES)
    assert tuple(a.shape) == tuple(x.shape), \
        f"the step must keep the sample shape {tuple(x.shape)}, got {tuple(a.shape)}"
    assert torch.equal(a, b), (
        "two identical DDIM steps produced different results — eta = 0 means the "
        "update is fully deterministic, there is no sigma*z term")


def check_ddim_step_to_the_same_timestep_is_the_identity(ns):
    """sqrt(abar_t)*x0_pred + sqrt(1-abar_t)*eps rebuilds x_t exactly. Any model."""
    m = _NonlinearDenoiser()
    x, y = torch.randn(64, DIM), _labels(64, 1)
    for t in (5, 120, TOTAL - 1):
        out = ns.ddim_sample_step(m, x, t, t, ABAR, y, 1.0, NUM_CLASSES)
        assert torch.allclose(out, x, atol=1e-4), (
            f"stepping from t={t} to t_prev={t} must return x_t unchanged: the "
            f"DDIM update predicts x_0 and then re-noises it to the t_prev level, "
            f"so with t_prev == t the two operations cancel; max deviation "
            f"{float((out - x).abs().max()):.5f}")


def check_ddim_step_matches_the_eta_zero_update(ns):
    m = _ConstDenoiser(0.4)
    x, y = torch.randn(64, DIM), _labels(64, 2)
    for t, t_prev in ((250, 200), (120, 60), (30, 0)):
        got = ns.ddim_sample_step(m, x, t, t_prev, ABAR, y, 1.0, NUM_CLASSES)
        expected = _ddim(x, t, t_prev, torch.full_like(x, 0.4))
        assert torch.allclose(got, expected, atol=1e-4), (
            f"the deterministic DDIM update from t={t} to t_prev={t_prev} is "
            f"sqrt(abar_prev)*x0_pred + sqrt(1-abar_prev)*eps with "
            f"x0_pred = (x_t - sqrt(1-abar_t)*eps)/sqrt(abar_t); max deviation "
            f"{float((got - expected).abs().max()):.5f}")


def check_ddim_final_step_returns_predicted_x0(ns):
    m = _ConstDenoiser(-0.3)
    x, y = torch.randn(64, DIM), _labels(64, 0)
    got = ns.ddim_sample_step(m, x, 60, -1, ABAR, y, 1.0, NUM_CLASSES)
    expected = _ddim(x, 60, -1, torch.full_like(x, -0.3))
    assert torch.allclose(got, expected, atol=1e-4), (
        "t_prev = -1 marks the final step: return the predicted x_0, "
        "(x_t - sqrt(1-abar_t)*eps)/sqrt(abar_t); max deviation "
        f"{float((got - expected).abs().max()):.5f}")


def check_ddim_step_applies_guidance(ns):
    m = _LabelDenoiser()
    x, y = torch.randn(32, DIM), _labels(32, 1)
    got = ns.ddim_sample_step(m, x, 150, 100, ABAR, y, 3.0, NUM_CLASSES)
    eps = torch.full_like(x, NUM_CLASSES + 3.0 * (1 - NUM_CLASSES))
    expected = _ddim(x, 150, 100, eps)
    assert torch.allclose(got, expected, atol=1e-4), (
        "with guidance_scale = 3 the step must use the guided noise "
        f"{float(eps.flatten()[0]):.1f} (= {NUM_CLASSES} + 3*(1 - {NUM_CLASSES})), "
        f"not the raw conditional prediction; max deviation "
        f"{float((got - expected).abs().max()):.5f}")


# --------------------------------------------------------------------------
# ddim_sample_loop
# --------------------------------------------------------------------------
def _loop(ns, model, n=64, steps=10, gs=1.0, label=1, seed=0):
    torch.manual_seed(seed)
    return ns.ddim_sample_loop(model, (n, DIM), ABAR, _labels(n, label),
                               num_steps=steps, guidance_scale=gs,
                               num_classes=NUM_CLASSES, total_timesteps=TOTAL)


def check_ddim_loop_shape(ns):
    out = _loop(ns, _NonlinearDenoiser(), n=17, steps=10)
    assert tuple(out.shape) == (17, DIM), \
        f"ddim_sample_loop must return the requested shape (17, {DIM}), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "generated samples contain NaN/Inf"


def check_ddim_loop_is_reproducible(ns):
    """eta = 0: once x_T is fixed the whole trajectory is fixed."""
    m = _NonlinearDenoiser()
    a = _loop(ns, m, steps=10, seed=7)
    b = _loop(ns, m, steps=10, seed=7)
    assert torch.equal(a, b), (
        "two runs from the same seed produced different samples — with eta = 0 the "
        "only randomness allowed is the initial x_T ~ N(0, I); the sampling steps "
        "themselves must not draw noise")


def check_ddim_loop_uses_a_decreasing_subsequence_of_timesteps(ns):
    m = _RecordingDenoiser()
    steps = 10
    torch.manual_seed(0)
    ns.ddim_sample_loop(m, (4, DIM), ABAR, _labels(4, 1), num_steps=steps,
                        guidance_scale=1.0, num_classes=NUM_CLASSES,
                        total_timesteps=TOTAL)
    seen = [t for i, t in enumerate(m.seen) if i == 0 or t != m.seen[i - 1]]
    assert len(seen) == steps, (
        f"asking for num_steps={steps} must visit exactly {steps} timesteps; the "
        f"denoiser was called at {len(seen)} distinct timesteps ({seen})")
    assert all(b < a for a, b in zip(seen, seen[1:])), \
        f"DDIM walks the timesteps backwards, so they must strictly decrease; got {seen}"
    assert 0 <= min(seen) and max(seen) < TOTAL, (
        f"timesteps must index the schedule, i.e. lie in [0, {TOTAL}); got range "
        f"[{min(seen)}, {max(seen)}]")
    stride = TOTAL // steps
    assert seen[0] >= TOTAL - 2 * stride, (
        f"sampling must start near the noisiest end of the schedule (~{TOTAL}), "
        f"it started at t={seen[0]}")
    assert seen[-1] <= stride, \
        f"sampling must finish at (or next to) t=0, it stopped at t={seen[-1]}"


def check_ddim_loop_is_class_conditional(ns):
    m = _LabelDenoiser()
    a = _loop(ns, m, label=0, seed=3)
    b = _loop(ns, m, label=3, seed=3)
    assert not torch.allclose(a, b, atol=1e-4), (
        "same seed, different class label, identical samples — class_label is not "
        "being threaded down into the denoiser")


def check_ddim_loop_applies_guidance(ns):
    m = _LabelDenoiser()
    a = _loop(ns, m, gs=1.0, seed=11)
    b = _loop(ns, m, gs=5.0, seed=11)
    assert not torch.allclose(a, b, atol=1e-4), (
        "same seed, guidance_scale 1.0 vs 5.0, identical samples — guidance_scale "
        "is not reaching guided_predict inside the loop")


def check_ddim_loop_step_count_changes_the_trajectory(ns):
    m = _NonlinearDenoiser()
    a = _loop(ns, m, steps=5, seed=13)
    b = _loop(ns, m, steps=50, seed=13)
    assert not torch.allclose(a, b, atol=1e-4), (
        "5-step and 50-step sampling from the same x_T gave the same answer — "
        "num_steps is being ignored when the timestep subsequence is built")


CHECKS = [
    check_conditional_denoiser_shape,
    check_conditional_denoiser_accepts_the_null_class,
    check_conditional_denoiser_is_deterministic_in_eval,
    check_guided_predict_matches_the_cfg_blend,
    check_guidance_scale_one_is_the_plain_conditional,
    check_guidance_scale_zero_is_unconditional,
    check_ddim_step_is_deterministic,
    check_ddim_step_to_the_same_timestep_is_the_identity,
    check_ddim_step_matches_the_eta_zero_update,
    check_ddim_final_step_returns_predicted_x0,
    check_ddim_step_applies_guidance,
    check_ddim_loop_shape,
    check_ddim_loop_is_reproducible,
    check_ddim_loop_uses_a_decreasing_subsequence_of_timesteps,
    check_ddim_loop_is_class_conditional,
    check_ddim_loop_applies_guidance,
    check_ddim_loop_step_count_changes_the_trajectory,
]
