"""ddpm — noise schedule, forward diffusion, denoiser, and the reverse sampler.

The reverse process is graded with a stub denoiser that is *exactly* optimal for
standard-normal data: for x_0 ~ N(0, I) the posterior mean of the noise is
E[eps | x_t] = sqrt(1 - abar_t) * x_t. Plugged into a correct DDPM step this
gives x_{t-1} = sqrt(alpha_t) * x_t + sqrt(beta_t) * z, which is exactly
variance preserving — so a correct p_sample / p_sample_loop must turn N(0, I)
noise back into N(0, I) samples, and any wrong coefficient (a missing
1/sqrt(alpha_t), sigma = beta instead of sqrt(beta)) visibly shrinks or blows up
the spread. No reference sampler is involved.
"""
import torch
import torch.nn as nn

from torchleet.runner import Skip

ENTRIES = ["linear_beta_schedule", "q_sample", "SimpleDenoiser",
           "p_sample", "p_sample_loop"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "linear_beta_schedule is torch.linspace(beta_start, beta_end, timesteps) — "
    "small betas first, so early steps corrupt slowly.",
    "q_sample is the closed form of the forward process: "
    "x_t = sqrt(abar_t)*x_0 + sqrt(1-abar_t)*noise, with abar gathered per sample "
    "from t (each row of the batch has its own timestep).",
    "p_sample: mu = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-abar_t) * eps_pred), "
    "then add sqrt(beta_t)*z for t > 0 and nothing at t = 0. p_sample_loop starts "
    "at x_T ~ N(0, I) and calls p_sample for t = T-1 ... 0.",
]

T = 60
DIM = 2


def _schedule(ns, timesteps=T, beta_start=1e-4, beta_end=0.02):
    betas = ns.linear_beta_schedule(timesteps, beta_start, beta_end)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


class _OptimalDenoiser(nn.Module):
    """The exact E[eps | x_t] when the data distribution is N(0, I)."""

    def __init__(self, alphas_cumprod):
        super().__init__()
        self.coef = torch.sqrt(1.0 - alphas_cumprod)

    def forward(self, x, t):
        c = self.coef[t.long()].reshape(-1, *([1] * (x.dim() - 1)))
        return c * x


class _ZeroDenoiser(nn.Module):
    """Predicts no noise, so each reverse step is a pure known rescaling."""

    def forward(self, x, t):
        return torch.zeros_like(x)


class _RecordingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x, t):
        self.seen.append(int(t.reshape(-1)[0]))
        return torch.zeros_like(x)


def _numpy_skip(e):
    msg = str(e)
    if "numpy" in msg or "'np'" in msg:
        raise Skip("this notebook's timestep embedding imports numpy — "
                   "pip install numpy to check SimpleDenoiser")
    raise e


def _denoiser(ns):
    for kw in (dict(data_dim=DIM, hidden_dim=32, time_dim=16), dict(), ):
        try:
            return ns.SimpleDenoiser(**kw)
        except TypeError:
            continue
        except (NameError, ImportError) as e:
            _numpy_skip(e)
    raise AssertionError(
        "could not build SimpleDenoiser(data_dim=..., hidden_dim=..., time_dim=...)")


# --------------------------------------------------------------------------
# linear_beta_schedule
# --------------------------------------------------------------------------
def check_beta_schedule_endpoints_and_monotonicity(ns):
    betas = ns.linear_beta_schedule(T, 1e-4, 0.02)
    assert tuple(betas.shape) == (T,), \
        f"the schedule should have one beta per timestep {(T,)}, got {tuple(betas.shape)}"
    assert abs(float(betas[0]) - 1e-4) < 1e-9, \
        f"betas[0] should be beta_start=1e-4, got {float(betas[0]):.8f}"
    assert abs(float(betas[-1]) - 0.02) < 1e-9, \
        f"betas[-1] should be beta_end=0.02, got {float(betas[-1]):.8f}"
    assert bool((betas[1:] > betas[:-1]).all()), \
        "betas must increase: later steps add more noise"
    assert bool(((betas > 0) & (betas < 1)).all()), (
        f"every beta is a variance fraction and must lie in (0, 1), got range "
        f"[{float(betas.min()):.5f}, {float(betas.max()):.5f}]")
    step = betas[1:] - betas[:-1]
    assert torch.allclose(step, step[0].expand_as(step), atol=1e-9), \
        "a *linear* schedule has a constant gap between consecutive betas"


def check_alphas_cumprod_decays_to_almost_zero_signal(ns):
    betas, alphas, abar = _schedule(ns, 400, 1e-4, 0.02)
    assert bool((abar[1:] < abar[:-1]).all()), \
        "alphas_cumprod must decrease — the surviving signal only shrinks"
    assert float(abar[0]) > 0.999, \
        f"almost all signal must survive the first step, got abar[0]={float(abar[0]):.5f}"
    assert float(abar[-1]) < 0.05, (
        f"after the full schedule x_T should be nearly pure noise "
        f"(abar_T << 1), got abar[-1]={float(abar[-1]):.5f}")


# --------------------------------------------------------------------------
# q_sample
# --------------------------------------------------------------------------
def check_q_sample_matches_the_forward_closed_form(ns):
    betas, alphas, abar = _schedule(ns)
    sa, soma = torch.sqrt(abar), torch.sqrt(1.0 - abar)
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(8, DIM, generator=g)
    noise = torch.randn(8, DIM, generator=g)
    t = torch.full((8,), 30, dtype=torch.long)
    got = ns.q_sample(x0, t, noise, sa, soma)
    expected = sa[30] * x0 + soma[30] * noise
    assert tuple(got.shape) == tuple(x0.shape), \
        f"q_sample must keep the data shape {tuple(x0.shape)}, got {tuple(got.shape)}"
    assert torch.allclose(got, expected, atol=1e-5), (
        "q(x_t | x_0) = sqrt(abar_t)*x_0 + sqrt(1-abar_t)*noise; max deviation "
        f"{float((got - expected).abs().max()):.5f}")


def check_q_sample_uses_each_row_own_timestep(ns):
    betas, alphas, abar = _schedule(ns)
    sa, soma = torch.sqrt(abar), torch.sqrt(1.0 - abar)
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(4, DIM, generator=g)
    noise = torch.randn(4, DIM, generator=g)
    t = torch.tensor([0, 10, 30, T - 1])
    got = ns.q_sample(x0, t, noise, sa, soma)
    expected = sa[t].unsqueeze(-1) * x0 + soma[t].unsqueeze(-1) * noise
    assert torch.allclose(got, expected, atol=1e-5), (
        "each row of the batch carries its own timestep, so gather the schedule "
        "with t and broadcast per row (not t[0] for the whole batch); max deviation "
        f"{float((got - expected).abs().max()):.5f}")


def check_q_sample_is_variance_preserving(ns):
    """With x_0 and noise both standard normal, x_t stays unit variance."""
    betas, alphas, abar = _schedule(ns)
    sa, soma = torch.sqrt(abar), torch.sqrt(1.0 - abar)
    g = torch.Generator().manual_seed(2)
    x0 = torch.randn(20000, DIM, generator=g)
    noise = torch.randn(20000, DIM, generator=g)
    for step in (0, T // 2, T - 1):
        t = torch.full((20000,), step, dtype=torch.long)
        xt = ns.q_sample(x0, t, noise, sa, soma)
        s = float(xt.std())
        assert abs(s - 1.0) < 0.05, (
            f"the forward process is variance preserving: with x_0 ~ N(0,1) and "
            f"unit noise, x_{step} should still have std ~1, got {s:.3f}")


# --------------------------------------------------------------------------
# SimpleDenoiser
# --------------------------------------------------------------------------
def check_denoiser_predicts_noise_shaped_like_the_input(ns):
    m = _denoiser(ns)
    x = torch.randn(5, DIM)
    t = torch.randint(0, T, (5,))
    try:
        out = m(x, t)
    except (NameError, ImportError) as e:
        _numpy_skip(e)
    assert tuple(out.shape) == (5, DIM), (
        f"the denoiser predicts the noise added to x, so the output must match the "
        f"input shape {(5, DIM)}, got {tuple(out.shape)}")
    assert torch.isfinite(out).all(), "denoiser output contains NaN/Inf"


def check_denoiser_is_conditioned_on_the_timestep(ns):
    m = _denoiser(ns)
    m.eval()
    x = torch.randn(5, DIM)
    try:
        with torch.no_grad():
            early = m(x, torch.zeros(5, dtype=torch.long))
            late = m(x, torch.full((5,), T - 1, dtype=torch.long))
    except (NameError, ImportError) as e:
        _numpy_skip(e)
    assert not torch.allclose(early, late, atol=1e-6), (
        "the same x at t=0 and t=T-1 produced the same prediction — the timestep "
        "embedding is not reaching the network, so it cannot know how much noise "
        "to remove")


def check_denoiser_gradients_flow(ns):
    m = _denoiser(ns)
    x = torch.randn(5, DIM, requires_grad=True)
    try:
        out = m(x, torch.randint(0, T, (5,)))
    except (NameError, ImportError) as e:
        _numpy_skip(e)
    out.sum().backward()
    assert x.grad is not None and bool((x.grad != 0).any()), \
        "no gradient reached the input — the denoiser is not differentiable end to end"
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(bool((gr != 0).any()) for gr in grads), \
        "no denoiser parameter received a gradient"


# --------------------------------------------------------------------------
# p_sample
# --------------------------------------------------------------------------
def check_p_sample_final_step_is_deterministic(ns):
    """At t = 0 there is no more noise to add — the step returns the mean."""
    betas, alphas, abar = _schedule(ns)
    m = _ZeroDenoiser()
    x = torch.randn(64, DIM)
    a = ns.p_sample(m, x, 0, betas, alphas, abar)
    b = ns.p_sample(m, x, 0, betas, alphas, abar)
    assert tuple(a.shape) == tuple(x.shape), \
        f"p_sample must keep the shape {tuple(x.shape)}, got {tuple(a.shape)}"
    assert torch.allclose(a, b, atol=1e-6), \
        "the t=0 step must not add noise, so two calls must agree exactly"
    expected = x / torch.sqrt(alphas[0])
    assert torch.allclose(a, expected, atol=1e-5), (
        "with a denoiser that predicts zero noise, the t=0 step is just "
        "x_t / sqrt(alpha_0); max deviation "
        f"{float((a - expected).abs().max()):.6f} — check the 1/sqrt(alpha_t) factor")


def check_p_sample_adds_noise_before_the_final_step(ns):
    betas, alphas, abar = _schedule(ns)
    m = _OptimalDenoiser(abar)
    t = 30
    x = torch.randn(20000, DIM)
    a = ns.p_sample(m, x, t, betas, alphas, abar)
    b = ns.p_sample(m, x, t, betas, alphas, abar)
    assert not torch.allclose(a, b, atol=1e-6), (
        f"two p_sample calls at t={t} came out identical — every step except t=0 "
        "must add sigma*z with fresh Gaussian noise")
    mean = torch.sqrt(alphas[t]) * x    # what the optimal denoiser implies
    resid = float((a - mean).std())
    want = float(torch.sqrt(betas[t]))
    assert abs(resid - want) < 0.15 * want, (
        f"the noise added at step t={t} should have std sqrt(beta_t) = {want:.4f}, "
        f"measured {resid:.4f} around the posterior mean")


def check_p_sample_preserves_unit_variance(ns):
    """Under the exactly-optimal denoiser for N(0,I) data, every reverse step is
    variance preserving — so chaining them all must still land on N(0,1)."""
    betas, alphas, abar = _schedule(ns)
    m = _OptimalDenoiser(abar)
    x = torch.randn(8000, DIM)
    one = ns.p_sample(m, x, T // 2, betas, alphas, abar)
    assert abs(float(one.std()) - 1.0) < 0.05 and abs(float(one.mean())) < 0.05, (
        "fed the exactly-optimal denoiser for standard-normal data, one reverse "
        f"step must map N(0,1) to N(0,1); got mean {float(one.mean()):.3f}, std "
        f"{float(one.std()):.3f}")
    for t in reversed(range(T)):
        x = ns.p_sample(m, x, t, betas, alphas, abar)
    s, mu = float(x.std()), float(x.mean())
    assert abs(s - 1.0) < 0.08 and abs(mu) < 0.06, (
        f"chaining p_sample from t={T - 1} down to t=0 under that same optimal "
        f"denoiser must still give N(0,1); got mean {mu:.3f}, std {s:.3f}. A std "
        f"below 1 compounding like this means the 1/sqrt(alpha_t) rescale is "
        f"missing; above 1 means sigma is too large (it should be sqrt(beta_t)).")


# --------------------------------------------------------------------------
# p_sample_loop
# --------------------------------------------------------------------------
def check_p_sample_loop_shape(ns):
    betas, alphas, abar = _schedule(ns)
    out = ns.p_sample_loop(_ZeroDenoiser(), (17, DIM), T, betas, alphas, abar)
    assert tuple(out.shape) == (17, DIM), \
        f"p_sample_loop must return samples of the requested shape (17, {DIM}), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "generated samples contain NaN/Inf"


def check_p_sample_loop_visits_every_timestep_backwards(ns):
    betas, alphas, abar = _schedule(ns)
    m = _RecordingDenoiser()
    ns.p_sample_loop(m, (4, DIM), T, betas, alphas, abar)
    assert len(m.seen) == T, (
        f"the loop should call the denoiser once per timestep ({T} times), it "
        f"called it {len(m.seen)} times")
    assert m.seen == list(reversed(range(T))), (
        f"sampling runs backwards from t=T-1 down to t=0; the timesteps actually "
        f"visited start {m.seen[:3]} and end {m.seen[-3:]}")


def check_p_sample_loop_starts_from_pure_noise(ns):
    """One step of a beta=0.75 schedule turns x_T ~ N(0,1) into std 2."""
    betas = torch.tensor([0.75])
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    out = ns.p_sample_loop(_ZeroDenoiser(), (30000, DIM), 1, betas, alphas, abar)
    s = float(out.std())
    assert abs(s - 2.0) < 0.1, (
        "with a single timestep, beta=0.75 and a zero-noise denoiser, the loop "
        "should draw x_T ~ N(0,1) and return x_T / sqrt(alpha_0) = 2*x_T, i.e. "
        f"std 2.0; got std {s:.3f} (std ~1.0 means no step was applied, std ~0 "
        "means the loop started from zeros instead of Gaussian noise)")


def check_p_sample_loop_recovers_the_data_distribution(ns):
    """The whole point: exact denoiser in, the training distribution out."""
    betas, alphas, abar = _schedule(ns)
    out = ns.p_sample_loop(_OptimalDenoiser(abar), (8000, DIM), T, betas, alphas, abar)
    mu, s = float(out.mean()), float(out.std())
    assert abs(mu) < 0.06, \
        f"samples should be centred like the N(0,1) data the denoiser was optimal for, got mean {mu:.3f}"
    assert abs(s - 1.0) < 0.08, (
        f"given the exactly-optimal denoiser for N(0,1) data, the full reverse "
        f"chain must reproduce N(0,1); got std {s:.3f} after {T} steps. Compounding "
        f"error like this comes from a wrong coefficient in the posterior mean or "
        f"from sigma != sqrt(beta_t).")


CHECKS = [
    check_beta_schedule_endpoints_and_monotonicity,
    check_alphas_cumprod_decays_to_almost_zero_signal,
    check_q_sample_matches_the_forward_closed_form,
    check_q_sample_uses_each_row_own_timestep,
    check_q_sample_is_variance_preserving,
    check_denoiser_predicts_noise_shaped_like_the_input,
    check_denoiser_is_conditioned_on_the_timestep,
    check_denoiser_gradients_flow,
    check_p_sample_final_step_is_deterministic,
    check_p_sample_adds_noise_before_the_final_step,
    check_p_sample_preserves_unit_variance,
    check_p_sample_loop_shape,
    check_p_sample_loop_visits_every_timestep_backwards,
    check_p_sample_loop_starts_from_pure_noise,
    check_p_sample_loop_recovers_the_data_distribution,
]
