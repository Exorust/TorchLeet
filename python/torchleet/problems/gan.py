"""gan — a generator and a discriminator that can be trained against each other.

The architectures are free, so nothing is compared against a reference. What is
checked is the contract the adversarial loop depends on: the generator turns
latent noise into data-shaped samples in the Tanh range, the discriminator turns
a sample into one probability, and gradients survive the G -> D composition
(the classic mistake is detaching the generator when training it).
"""
import torch
import torch.nn as nn

ENTRIES = ["Generator", "Discriminator"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Generator(input_dim, output_dim) maps a latent vector to one fake sample; "
    "Discriminator(input_dim) maps a sample to a single number.",
    "End the generator with Tanh so samples live in [-1, 1], and the "
    "discriminator with Sigmoid so its output is a probability BCELoss can take.",
    "When you train the generator do NOT detach the fake batch — the gradient has "
    "to flow back through the discriminator into the generator.",
]

LATENT, DATA = 10, 1


def _pair(ns, latent=LATENT, data=DATA, seed=0, eval_mode=False):
    torch.manual_seed(seed)
    g, d = ns.Generator(latent, data), ns.Discriminator(data)
    if eval_mode:
        # Shape and range checks run at inference time: a DCGAN-style generator
        # with BatchNorm legitimately refuses a batch of 1 while training.
        g.eval()
        d.eval()
    return g, d


def check_generator_output_shape(ns):
    for latent, data in ((10, 1), (5, 3), (16, 8)):
        torch.manual_seed(0)
        g = ns.Generator(latent, data)
        g.eval()
        out = g(torch.randn(4, latent))
        assert tuple(out.shape) == (4, data), (
            f"Generator({latent}, {data}) on a (4, {latent}) latent batch gave "
            f"{tuple(out.shape)}, expected {(4, data)}")


def check_generator_output_in_tanh_range(ns):
    g, _ = _pair(ns, eval_mode=True)
    with torch.no_grad():
        out = g(torch.randn(64, LATENT) * 5)
    assert float(out.min()) >= -1.0 - 1e-5 and float(out.max()) <= 1.0 + 1e-5, (
        f"generator samples span [{float(out.min()):.3f}, {float(out.max()):.3f}] "
        "but the real data lives in [-1, 1] — finish the generator with Tanh")


def check_discriminator_outputs_one_probability(ns):
    _, d = _pair(ns, eval_mode=True)
    with torch.no_grad():
        out = d(torch.randn(6, DATA) * 5)
    assert tuple(out.shape) == (6, 1), (
        f"discriminator gave {tuple(out.shape)} for a (6, {DATA}) batch, expected "
        "(6, 1): one score per sample")
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, (
        f"discriminator outputs span [{float(out.min()):.3f}, "
        f"{float(out.max()):.3f}] — BCELoss needs a probability, so finish with "
        "Sigmoid (or switch to BCEWithLogitsLoss)")


def check_batch_agnostic(ns):
    g, d = _pair(ns, eval_mode=True)
    for b in (1, 3, 32):
        fake = g(torch.randn(b, LATENT))
        assert tuple(fake.shape) == (b, DATA), \
            f"generator: batch {b} gave {tuple(fake.shape)}, expected {(b, DATA)}"
        score = d(fake)
        assert tuple(score.shape) == (b, 1), \
            f"discriminator: batch {b} gave {tuple(score.shape)}, expected {(b, 1)}"


def check_generator_gradients_flow_through_discriminator(ns):
    """The generator only ever learns through the discriminator's opinion."""
    g, d = _pair(ns)
    z = torch.randn(16, LATENT)
    loss = nn.BCELoss()(d(g(z)), torch.ones(16, 1))
    loss.backward()
    dead = [n for n, p in g.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached the generator parameters: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — either they are off the forward path "
        "or the fake batch was detached before the discriminator saw it")


def check_discriminator_gradients_flow(ns):
    _, d = _pair(ns)
    nn.BCELoss()(d(torch.randn(16, DATA)), torch.ones(16, 1)).backward()
    dead = [n for n, p in d.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — not on the discriminator's forward path")


def check_discriminator_can_separate(ns):
    """Overfit real-vs-fake on two obviously different clouds; loss must fall.

    Catches a discriminator that is wired together but cannot train — saturated
    Sigmoid, dead ReLU, or an output that ignores its input.
    """
    torch.manual_seed(0)
    _, d = _pair(ns)
    real = torch.rand(32, DATA) * 0.5 + 0.5      # samples near +1
    fake = torch.rand(32, DATA) * 0.5 - 1.0      # samples near -1
    x = torch.cat([real, fake])
    y = torch.cat([torch.ones(32, 1), torch.zeros(32, 1)])
    opt = torch.optim.Adam(d.parameters(), lr=1e-3)
    lossf = nn.BCELoss()
    first = lossf(d(x), y).item()
    loss = None
    for _ in range(60):
        opt.zero_grad()
        loss = lossf(d(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first, (
        f"the discriminator could not learn to tell two well-separated clouds "
        f"apart over 60 steps ({first:.4f} -> {loss.item():.4f})")


CHECKS = [
    check_generator_output_shape,
    check_generator_output_in_tanh_range,
    check_discriminator_outputs_one_probability,
    check_batch_agnostic,
    check_generator_gradients_flow_through_discriminator,
    check_discriminator_gradients_flow,
    check_discriminator_can_separate,
]
