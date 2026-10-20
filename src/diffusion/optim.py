"""Adam, and an exponential moving average of the weights for sampling."""

from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 2e-4, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray], lr: float | None = None) -> None:
        self.t += 1
        lr = self.lr if lr is None else lr
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            p -= lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)


def clip_grad_norm(grads: dict[str, np.ndarray], max_norm: float) -> float:
    norm = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values())))
    if norm > max_norm:
        for g in grads.values():
            g *= max_norm / norm
    return norm


class EMA:
    """Sampling from a running average of the weights gives visibly cleaner
    digits than the raw weights, which jitter step to step."""

    def __init__(self, params: dict[str, np.ndarray], decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.copy() for k, v in params.items()}

    def update(self, params: dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            self.shadow[k] *= self.decay
            self.shadow[k] += (1 - self.decay) * v
