"""mamba — the selective scan (S6) recurrence and the Mamba block around it.

The scan is graded with parameter settings that collapse the recurrence to
something a torch built-in already computes, so nothing here depends on how the
scan is written (python loop, associative scan, parallel prefix — all fine):

  * delta = 0 freezes the state, so the output must be exactly the skip term D*x;
  * A = 0, delta = B = C = 1, D = 0 turns the recurrence into a running sum, so
    the output must equal torch.cumsum(x, dim=1);
  * changing the last timestep must not change any earlier output — a scan that
    peeks at the future fails this, and so does one that runs backwards.
"""
import torch

ENTRIES = ["selective_scan", "MambaBlock"]
DEVICE = "cpu"
EXTRAS = []          # the notebook plots with matplotlib/numpy; these checks do not

HINTS = [
    "Discretize first: A_bar = exp(delta[..., None] * A) and B_bar = "
    "delta[..., None] * B[:, :, None, :], both (B, L, D, N).",
    "Then run the recurrence over time with h of shape (B, D, N): "
    "h = A_bar[:, t] * h + B_bar[:, t] * x[:, t, :, None], y_t = (h * C[:, t, None, :]).sum(-1).",
    "Two things are easy to lose: the skip connection y = y + D * x, and causality "
    "in MambaBlock — Conv1d(padding=d_conv-1) must be truncated back to L so no "
    "position sees the future.",
]

BATCH, LEN, D_INNER = 2, 7, 5


def _rand(*shape, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape)


def check_scan_output_shape(ns):
    x = _rand(BATCH, LEN, D_INNER, seed=0)
    n = 4
    y = ns.selective_scan(x, torch.rand(BATCH, LEN, D_INNER) + 0.1,
                          -torch.rand(D_INNER, n), torch.randn(BATCH, LEN, n),
                          torch.randn(BATCH, LEN, n), torch.randn(D_INNER))
    assert y is not None, "selective_scan returned None"
    assert tuple(y.shape) == (BATCH, LEN, D_INNER), \
        f"selective_scan should return (B, L, D) = {(BATCH, LEN, D_INNER)}, got {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "selective_scan produced NaN or Inf"


def check_zero_delta_leaves_only_the_skip(ns):
    """delta = 0 makes A_bar = 1 and B_bar = 0, so the state stays 0 and y = D * x."""
    x = _rand(BATCH, LEN, D_INNER, seed=1)
    n = 4
    D = torch.randn(D_INNER)
    y = ns.selective_scan(x, torch.zeros(BATCH, LEN, D_INNER), -torch.rand(D_INNER, n),
                          torch.randn(BATCH, LEN, n), torch.randn(BATCH, LEN, n), D)
    expected = D.view(1, 1, D_INNER) * x
    diff = (y - expected).abs().max().item()
    assert torch.allclose(y, expected, atol=1e-5), (
        f"with delta=0 nothing enters the state, so the output must be exactly the "
        f"skip connection D * x (max diff {diff:.2e}) — is `y = y + D * x` missing "
        f"or is the state being updated when delta is zero?")


def check_scan_is_a_running_sum_when_a_is_zero(ns):
    """A=0, delta=B=C=1, D=0, N=1 reduces the recurrence to h_t = h_{t-1} + x_t."""
    x = _rand(BATCH, LEN, D_INNER, seed=2)
    y = ns.selective_scan(x, torch.ones(BATCH, LEN, D_INNER), torch.zeros(D_INNER, 1),
                          torch.ones(BATCH, LEN, 1), torch.ones(BATCH, LEN, 1),
                          torch.zeros(D_INNER))
    expected = torch.cumsum(x, dim=1)
    diff = (y - expected).abs().max().item()
    assert torch.allclose(y, expected, atol=1e-4), (
        f"with A=0, delta=B=C=1 and D=0 the recurrence is h_t = h_{{t-1}} + x_t, so "
        f"the output must equal torch.cumsum(x, dim=1) (max diff {diff:.2e}). A "
        f"mismatch at t=0 means the state is not initialized to zeros; a mismatch "
        f"everywhere means the state is not carried across timesteps.")


def check_scan_is_causal(ns):
    """No output at time t may depend on inputs after t."""
    n = 3
    torch.manual_seed(3)
    x = torch.randn(BATCH, LEN, D_INNER)
    delta = torch.rand(BATCH, LEN, D_INNER) + 0.1
    A = -torch.rand(D_INNER, n)
    B = torch.randn(BATCH, LEN, n)
    C = torch.randn(BATCH, LEN, n)
    D = torch.randn(D_INNER)
    y = ns.selective_scan(x, delta, A, B, C, D)

    x2, delta2, B2, C2 = x.clone(), delta.clone(), B.clone(), C.clone()
    x2[:, -1] = torch.randn(BATCH, D_INNER)
    delta2[:, -1] = torch.rand(BATCH, D_INNER) + 0.1
    B2[:, -1] = torch.randn(BATCH, n)
    C2[:, -1] = torch.randn(BATCH, n)
    y2 = ns.selective_scan(x2, delta2, A, B2, C2, D)

    diff = (y[:, :-1] - y2[:, :-1]).abs().max().item()
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-6), (
        f"changing only the last timestep changed earlier outputs (max diff "
        f"{diff:.2e}) — the scan must run forwards in time and never look ahead")


def check_scan_backpropagates(ns):
    n = 4
    torch.manual_seed(4)
    x = torch.randn(BATCH, LEN, D_INNER, requires_grad=True)
    delta = (torch.rand(BATCH, LEN, D_INNER) + 0.1).requires_grad_(True)
    A = (-torch.rand(D_INNER, n)).requires_grad_(True)
    D = torch.randn(D_INNER, requires_grad=True)
    y = ns.selective_scan(x, delta, A, torch.randn(BATCH, LEN, n),
                          torch.randn(BATCH, LEN, n), D)
    y.sum().backward()
    for name, t in (("x", x), ("delta", delta), ("A", A), ("D", D)):
        assert t.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(t.grad).all(), f"gradient w.r.t. {name} contains NaN/Inf"


def check_block_preserves_shape(ns):
    torch.manual_seed(5)
    blk = ns.MambaBlock(d_model=8, d_state=4, d_conv=4, expand=2)
    blk.eval()
    x = torch.randn(2, 9, 8)
    out = blk(x)
    assert tuple(out.shape) == (2, 9, 8), (
        f"MambaBlock must map (B, L, d_model) to (B, L, d_model): expected "
        f"{(2, 9, 8)}, got {tuple(out.shape)}")
    assert torch.isfinite(out).all(), "MambaBlock produced NaN or Inf"


def check_block_is_causal(ns):
    """The classic Mamba bug: a Conv1d whose padding lets position t see t+1."""
    torch.manual_seed(6)
    blk = ns.MambaBlock(d_model=8, d_state=4, d_conv=4, expand=2)
    blk.eval()
    x = torch.randn(2, 9, 8)
    x2 = x.clone()
    x2[:, -1] = torch.randn(2, 8)
    with torch.no_grad():
        out, out2 = blk(x), blk(x2)
    diff = (out[:, :-1] - out2[:, :-1]).abs().max().item()
    assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-5), (
        f"changing the last input token changed the outputs at earlier positions "
        f"(max diff {diff:.2e}) — the depthwise Conv1d is not causal: pad by "
        f"d_conv-1 and slice the result back to L so the future is dropped")


def check_block_trains_every_parameter(ns):
    torch.manual_seed(7)
    blk = ns.MambaBlock(d_model=8, d_state=4, d_conv=4, expand=2)
    out = blk(torch.randn(2, 6, 8))
    out.pow(2).sum().backward()
    dead = [n for n, p in blk.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, (
        f"no gradient reached {dead} — every projection, the state matrix A and "
        f"the skip D should take part in the forward pass")


CHECKS = [
    check_scan_output_shape,
    check_zero_delta_leaves_only_the_skip,
    check_scan_is_a_running_sum_when_a_is_zero,
    check_scan_is_causal,
    check_scan_backpropagates,
    check_block_preserves_shape,
    check_block_is_causal,
    check_block_trains_every_parameter,
]
