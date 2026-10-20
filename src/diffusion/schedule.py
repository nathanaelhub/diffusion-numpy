"""The diffusion process: noising images, and the two samplers that undo it.

Forward (Ho et al. 2020): x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, with
abar_t the running product of (1 - beta).

The network can be trained to predict eps (Ho et al.) or
v = sqrt(abar) eps - sqrt(1 - abar) x_0 (Salimans & Ho 2022). Recovering x_0
from an eps prediction divides by sqrt(abar), which near t = T multiplies
the error a hundredfold; v gives x_0 = sqrt(abar) x_t - sqrt(1 - abar) v with
no division, so it stays usable at high noise, where a digit's shape is
decided. Both samplers work from either target.
"""

from __future__ import annotations

import numpy as np


class Schedule:
    def __init__(self, T: int = 1000, kind: str = "cosine"):
        self.T = T
        if kind == "linear":
            betas = np.linspace(1e-4, 0.02, T)
        elif kind == "cosine":
            # Nichol & Dhariwal 2021: abar follows a squared cosine, so the
            # signal isn't destroyed in the first few hundred steps.
            s = 0.008
            f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2) ** 2
            betas = np.clip(1 - f[1:] / f[:-1], 0, 0.999)
        else:
            raise ValueError(f"unknown schedule {kind!r}")
        self.kind = kind
        self.betas = betas
        self.alphas = 1.0 - betas
        self.abar = np.cumprod(self.alphas)

    def noise(self, x0, t, eps):
        a = self.abar[t][:, None]
        return np.sqrt(a) * x0 + np.sqrt(1 - a) * eps


def target(x0, eps, abar_t, kind: str):
    """What the network is trained to output."""
    if kind == "eps":
        return eps
    a = abar_t[:, None]
    return np.sqrt(a) * eps - np.sqrt(1 - a) * x0


def eps_and_x0(out, x, ab: float, kind: str):
    """Network output at x_t -> (eps, x0 clipped to the data range)."""
    if kind == "eps":
        eps = out
        x0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)
    else:
        x0 = np.sqrt(ab) * x - np.sqrt(1 - ab) * out
        eps = np.sqrt(1 - ab) * x + np.sqrt(ab) * out
    return eps, np.clip(x0, -1, 1)


def _predict(net, x, t, y, guidance: float):
    """Network output, with classifier-free guidance when classes are given:
    out = out_uncond + w (out_cond - out_uncond). Guidance is linear, so it
    applies to an eps or a v output alike."""
    tt = np.full(len(x), t)
    if y is None or guidance == 1.0:
        return net.forward(x, tt, y)
    null = np.full(len(x), net.config["n_classes"])
    both = net.forward(np.concatenate([x, x]), np.concatenate([tt, tt]), np.concatenate([y, null]))
    cond, unc = both[: len(x)], both[len(x):]
    return unc + guidance * (cond - unc)


def ddpm_sample(net, sched: Schedule, n: int, rng, y=None, guidance: float = 1.0, dim: int = 784,
                pred: str = "eps"):
    """Ancestral sampling, one network call per step (T calls). The posterior
    mean is taken from the clipped x0 estimate, the usual guard against
    samples drifting out of range."""
    x = rng.standard_normal((n, dim)).astype(net.dtype)
    for t in range(sched.T - 1, -1, -1):
        a, ab, b = sched.alphas[t], sched.abar[t], sched.betas[t]
        _, x0 = eps_and_x0(_predict(net, x, t, y, guidance), x, ab, pred)
        ab_prev = sched.abar[t - 1] if t > 0 else 1.0
        mean = (np.sqrt(ab_prev) * b / (1 - ab)) * x0 + (np.sqrt(a) * (1 - ab_prev) / (1 - ab)) * x
        if t > 0:
            var = b * (1 - ab_prev) / (1 - ab)  # posterior variance
            x = mean + np.sqrt(var) * rng.standard_normal(x.shape)
        else:
            x = mean
        x = x.astype(net.dtype)
    return x


def ddim_sample(net, sched: Schedule, n: int, rng, steps: int = 50, y=None, guidance: float = 1.0,
                dim: int = 784, eta: float = 0.0, pred: str = "eps", t_min: int = 0):
    """DDIM (Song et al. 2021): deterministic for eta = 0, ``steps`` network calls.

    ``t_min`` > 0 stops early and returns the model's clean-image estimate at
    that step, skipping the lowest-noise steps (see the README: this MLP's
    predictions there add speckle rather than remove it).
    """
    ts = np.linspace(sched.T - 1, t_min, steps).round().astype(int)
    x = rng.standard_normal((n, dim)).astype(net.dtype)
    for i, t in enumerate(ts):
        ab = sched.abar[t]
        _, x0 = eps_and_x0(_predict(net, x, t, y, guidance), x, ab, pred)
        if i == len(ts) - 1:
            x = x0
            break
        ab_next = sched.abar[ts[i + 1]]
        sigma = eta * np.sqrt((1 - ab_next) / (1 - ab) * (1 - ab / ab_next))
        # Re-derive eps from the clipped x0 so the two stay consistent.
        eps = (x - np.sqrt(ab) * x0) / np.sqrt(1 - ab)
        x = (np.sqrt(ab_next) * x0 + np.sqrt(1 - ab_next - sigma ** 2) * eps
             + sigma * rng.standard_normal(x.shape)).astype(net.dtype)
    return x
