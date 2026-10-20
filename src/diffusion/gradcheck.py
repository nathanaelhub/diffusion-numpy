"""Check the hand-derived backward pass against finite differences.

    python -m diffusion.gradcheck
"""

from __future__ import annotations

import numpy as np

from .nn import Denoiser


def loss_and_grad(net: Denoiser, x, t, y, target):
    """L = sum((net(x) - target)^2) and dL/dout.

    Training uses the mean; the sum has the same gradient up to a constant
    and keeps small gradients well above finite-difference round-off
    (about 1e-10 here), which a mean pushes them toward.
    """
    out = net.forward(x, t, y)
    diff = out - target
    return float((diff ** 2).sum()), 2.0 * diff


ATOL, RTOL = 1e-8, 1e-5


def check(n_classes: int = 10, seed: int = 0, eps: float = 1e-5, per_param: int = 6) -> float:
    """Worst |numeric - analytic| / (ATOL + RTOL |numeric|) over a sample of
    entries from every parameter; <= 1 means every entry passes.

    The tolerance is ``numpy.allclose``'s rule. A pure relative error breaks
    on near-zero gradients: an entry of 2.7e-6 with 7e-10 of finite-difference
    round-off reads as a 1e-4 "error".
    """
    rng = np.random.default_rng(seed)
    net = Denoiser(dim=12, width=16, blocks=2, temb_dim=8, n_classes=n_classes, seed=seed, dtype=np.float64)
    # Non-default LayerNorm gains and biases, so their gradients are exercised.
    for name, p in net.parameters().items():
        if name.endswith((".g", ".b")):
            p += rng.standard_normal(p.shape) * 0.3
    x = rng.standard_normal((5, 12))
    t = rng.integers(0, 1000, 5)
    y = rng.integers(0, n_classes + 1, 5) if n_classes else None
    target = rng.standard_normal((5, 12))

    net.zero_grad()
    _, dout = loss_and_grad(net, x, t, y, target)
    net.backward(dout)
    grads = net.gradients()

    worst = 0.0
    for name, p in net.parameters().items():
        flat = p.reshape(-1)
        for i in rng.choice(flat.size, min(per_param, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            lp, _ = loss_and_grad(net, x, t, y, target)
            flat[i] = old - eps
            lm, _ = loss_and_grad(net, x, t, y, target)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[name].reshape(-1)[i]
            worst = max(worst, abs(num - ana) / (ATOL + RTOL * abs(num)))
    return worst


if __name__ == "__main__":
    for n in (0, 10):
        e = check(n_classes=n)
        print(f"n_classes={n}: worst error / tolerance {e:.3f} -> {'PASS' if e <= 1 else 'FAIL'}")
