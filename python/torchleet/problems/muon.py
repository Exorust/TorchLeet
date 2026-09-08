"""muon — Muon built on torch.optim.Optimizer.

newton_schulz is graded as geometry: orthonormal rows/columns of the output,
scale invariance (orthogonalizing X or 7.3 * X gives the same answer — only
possible if the input is normalized first), and orthogonal matrices as fixed
points. Muon is graded by the *shape* of its matrix update: one step on a
wide or tall matrix must move it along a direction with orthonormal
columns/rows, so plain momentum without orthogonalization fails. Its 1D
fallback is pinned by the hand-derived first AdamW step, and the whole
optimizer must cut the loss of a small linear model at least in half.
"""
import torch

ENTRIES = ["newton_schulz", "MyMuon"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Subclass torch.optim.Optimizer and keep per-parameter state in "
    "self.state[p] — it starts as an empty dict, so initialize it on the "
    "first step. Do the whole update under @torch.no_grad() with in-place ops "
    "(mul_, add_, addcmul_, addcdiv_).",
    "Muon: buf = momentum*buf + grad (Nesterov uses grad + momentum*buf as "
    "the update), orthogonalize with your newton_schulz, scale by "
    "sqrt(max(m,n)/min(m,n)), and only for ndim >= 2 — 1D params get the "
    "AdamW fallback. newton_schulz: divide X by its Frobenius norm, then "
    "iterate X = a*X + b*(X@X.T)@X + c*(X@X.T)^2@X with a=15/8, b=-5/4, c=3/8.",
]


def _muon_matrix_step(ns, m, n, seed=7, **kw):
    g = torch.Generator().manual_seed(seed)
    p0 = torch.randn(m, n, generator=g)
    grad = torch.randn(m, n, generator=g)
    p = p0.clone().requires_grad_(True)
    defaults = dict(lr=0.05, momentum=0.9, nesterov=True, ns_steps=5,
                    weight_decay=0.0)
    defaults.update(kw)
    opt = ns.MyMuon([p], **defaults)
    p.grad = grad
    opt.step()
    return p0 - p.detach()  # = lr * scale * orthogonalized update


def check_newton_schulz_output_is_approximately_orthogonal(ns):
    # Only rectangular shapes: a square Gaussian matrix can have a near-zero
    # singular value, which no fixed number of iterations can pull up to 1.
    g = torch.Generator().manual_seed(4)
    for m, n in ((16, 32), (32, 16), (24, 8)):
        X = torch.randn(m, n, generator=g)
        O = ns.newton_schulz(X, steps=5)
        assert tuple(O.shape) == (m, n), (
            f"newton_schulz must keep the input shape {(m, n)}, "
            f"got {tuple(O.shape)}")
        gram = O @ O.T if m <= n else O.T @ O
        eye = torch.eye(gram.shape[0])
        dev = float((gram - eye).abs().max())
        assert torch.allclose(gram, eye, atol=0.15), (
            f"for a {m}x{n} input the {'rows' if m <= n else 'columns'} should "
            f"come out orthonormal, but the Gram matrix differs from the "
            f"identity by up to {dev:.3f}. The iteration is "
            "X = a*X + b*(X@X.T)@X + c*(X@X.T)^2@X with a=15/8, b=-5/4, c=3/8, "
            "starting from X normalized by its Frobenius norm")


def check_newton_schulz_is_scale_invariant(ns):
    """Orthogonalizing X or a multiple of X must give the same answer — only
    possible if the first step normalizes away the input scale."""
    g = torch.Generator().manual_seed(5)
    X = torch.randn(16, 32, generator=g)
    O1 = ns.newton_schulz(X, steps=5)
    O2 = ns.newton_schulz(7.3 * X, steps=5)
    dev = float((O1 - O2).abs().max())
    assert torch.allclose(O1, O2, atol=1e-4), (
        f"newton_schulz(X) and newton_schulz(7.3*X) differ by {dev:.4f} — the "
        "iteration only converges for spectral norm <= 1, so the input must be "
        "divided by its Frobenius norm (a cheap upper bound) BEFORE iterating")


def check_newton_schulz_keeps_orthogonal_matrices_fixed(ns):
    g = torch.Generator().manual_seed(6)
    Q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g))
    O = ns.newton_schulz(Q, steps=5)
    dev = float((O - Q).abs().max())
    assert torch.allclose(O, Q, atol=1e-2), (
        f"an already-orthogonal matrix is a fixed point of the iteration but "
        f"moved by {dev:.4f} — check the coefficients a=15/8, b=-5/4, c=3/8 "
        "(they satisfy a + b + c = 1, which is what makes singular value 1 "
        "stay put)")


def check_muon_matrix_step_moves_along_an_orthogonal_direction(ns):
    """The matrix update is lr*scale times an (approximately) orthogonal
    matrix, so its Gram matrix must be a multiple of the identity. Plain
    momentum without orthogonalization moves along the raw gradient, whose
    Gram matrix is random — and fails."""
    for m, n in ((16, 8), (8, 16)):
        delta = _muon_matrix_step(ns, m, n)
        gram = delta.T @ delta if m > n else delta @ delta.T
        diag = gram.diag()
        assert torch.allclose(diag, diag[0].expand_as(diag), rtol=0.2), (
            f"one Muon step on a {m}x{n} parameter must move it along an "
            f"orthogonal direction: all {'column' if m > n else 'row'} norms "
            f"of the update should be equal, but they range from "
            f"{float(diag.min()):.5f} to {float(diag.max()):.5f}. Did you "
            "orthogonalize the momentum with newton_schulz before applying it?")
        k = diag.shape[0]
        off = gram[~torch.eye(k, dtype=torch.bool)]
        ratio = float(off.abs().max() / diag.mean())
        assert ratio < 0.15, (
            f"the update's Gram matrix has off-diagonal entries {ratio:.2f}x "
            "the diagonal — the rows/columns are not orthogonal to each "
            "other, so the newton_schulz orthogonalization is missing or not "
            "converging")


def check_muon_falls_back_to_adamw_for_1d_params(ns):
    """1D parameters (biases, norm gains) cannot be orthogonalized — Muon
    falls back to AdamW for them, and the first AdamW step from zero state is
    -lr * sign(grad)."""
    g = torch.Generator().manual_seed(8)
    p0 = torch.randn(8, generator=g)
    grad = torch.randn(8, generator=g) * 5.0
    p = p0.clone().requires_grad_(True)
    opt = ns.MyMuon([p], lr=0.02, weight_decay=0.0,
                    adamw_betas=(0.9, 0.95), adamw_eps=1e-8)
    p.grad = grad
    opt.step()
    delta = p.detach() - p0
    expected = -0.02 * grad.sign()
    assert torch.allclose(delta, expected, atol=1e-6), (
        f"a 1D parameter should get the AdamW fallback: from zero state the "
        f"first step is -lr*sign(grad) = {float(expected[0]):.6f} on "
        f"coordinate 0, got {float(delta[0]):.6f}. newton_schulz is only for "
        "ndim >= 2; biases and norm gains need m/v moment updates with bias "
        "correction, exactly like AdamW")


def check_muon_reduces_loss_on_a_small_linear_model(ns):
    import torch.nn as nn
    g = torch.Generator().manual_seed(9)
    model = nn.Linear(16, 8)  # 2D weight (Muon branch) + 1D bias (AdamW branch)
    with torch.no_grad():
        model.weight.copy_(torch.randn(8, 16, generator=g))
        model.bias.copy_(torch.randn(8, generator=g))
    opt = ns.MyMuon(model.parameters(), lr=0.02, momentum=0.95,
                    weight_decay=1e-3)
    X = torch.randn(32, 16, generator=g)
    y = torch.randn(32, 8, generator=g)
    with torch.no_grad():
        first = float(((model(X) - y) ** 2).mean())
    for _ in range(200):
        loss = ((model(X) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = float(((model(X) - y) ** 2).mean())
    assert final < first * 0.5, (
        f"200 Muon steps should at least halve the MSE of a tiny linear "
        f"model: loss went {first:.4f} -> {final:.4f}. Check both branches — "
        "the 2D weight needs momentum + orthogonalization + the "
        "sqrt(max(m,n)/min(m,n)) scale, the 1D bias needs the AdamW fallback")


CHECKS = [
    check_newton_schulz_output_is_approximately_orthogonal,
    check_newton_schulz_is_scale_invariant,
    check_newton_schulz_keeps_orthogonal_matrices_fixed,
    check_muon_matrix_step_moves_along_an_orthogonal_direction,
    check_muon_falls_back_to_adamw_for_1d_params,
    check_muon_reduces_loss_on_a_small_linear_model,
]
