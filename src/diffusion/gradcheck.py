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


def tiny_model(arch: str, n_classes: int, seed: int):
    if arch == "unet":
        from .unet import UNet
        return UNet(dim=64, channels=(4, 8), groups=2, temb_dim=8, temb_width=8, n_classes=n_classes,
                    seed=seed, dtype=np.float64)
    return Denoiser(dim=12, width=16, blocks=2, temb_dim=8, n_classes=n_classes, seed=seed, dtype=np.float64)


def check(n_classes: int = 10, seed: int = 0, eps: float = 1e-5, per_param: int = 6, arch: str = "mlp") -> float:
    """Worst |numeric - analytic| / (ATOL + RTOL |numeric|) over a sample of
    entries from every parameter; <= 1 means every entry passes.

    The tolerance is ``numpy.allclose``'s rule. A pure relative error breaks
    on near-zero gradients: an entry of 2.7e-6 with 7e-10 of finite-difference
    round-off reads as a 1e-4 "error".
    """
    rng = np.random.default_rng(seed)
    net = tiny_model(arch, n_classes, seed)
    dim = net.config["dim"]
    # Non-default norm gains and biases, so their gradients are exercised.
    for name, p in net.parameters().items():
        if name.endswith((".g", ".b")):
            p += rng.standard_normal(p.shape) * 0.3
    x = rng.standard_normal((5, dim))
    t = rng.integers(0, 1000, 5)
    y = rng.integers(0, n_classes + 1, 5) if n_classes else None
    target = rng.standard_normal((5, dim))

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


def check_layer(layer, x: np.ndarray, seed: int = 0, eps: float = 1e-5, per_tensor: int = 12) -> float:
    """The same test for one layer: L = sum(r * layer(x)) for a fixed random r,
    checking dL/dx and every parameter gradient. Returns worst error / tolerance."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(layer.forward(x).shape)

    def loss():
        return float((r * layer.forward(x)).sum())

    if layer.params:
        layer.zero_grad()
    layer.forward(x)
    dx = layer.backward(r.copy())
    grads = {k: g.copy() for k, g in layer.grads.items()}
    tensors = [("x", x, dx)] + [(k, v, grads[k]) for k, v in layer.params.items()]
    worst = 0.0
    for _, arr, ana_all in tensors:
        flat, ana = arr.reshape(-1), ana_all.reshape(-1)
        for i in rng.choice(flat.size, min(per_tensor, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            lp = loss()
            flat[i] = old - eps
            lm = loss()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - ana[i]) / (ATOL + RTOL * abs(num)))
    return worst


if __name__ == "__main__":
    for arch in ("mlp", "unet"):
        for n in (0, 10):
            e = check(n_classes=n, arch=arch)
            print(f"{arch} n_classes={n}: worst error / tolerance {e:.3f} -> {'PASS' if e <= 1 else 'FAIL'}")
