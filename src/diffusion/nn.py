"""The noise-prediction network, with every gradient derived by hand.

An MLP over the flattened 28x28 image, conditioned on the diffusion step t
(and optionally the digit class) through an embedding that is added inside
every residual block:

    temb = Linear(SiLU(Linear(sinusoid(t)))) [+ class embedding]
    h    = Linear_in(x)
    h    = h + Linear_b(SiLU(Linear_a(LayerNorm(h)) + Linear_t(temb)))   x blocks
    eps  = Linear_out(SiLU(LayerNorm(h))) + Linear_skip(x)

The skip matters: much of the target is a linear function of x_t (for eps
prediction, eps is nearly x_t itself at high noise), and a 784-pixel linear
map can't pass through hidden layers narrower than 784. Measured with eps
prediction over 3,000 steps: width 512 reached a loss of 0.43 without the
skip and 0.20 with it; width 1024 without it, 0.16 at 2.4x the time.

Each layer keeps what its forward pass needs in ``self.cache`` and returns
input gradients from ``backward``; parameter gradients accumulate in
``self.grads`` under the same keys as ``self.params``.
"""

from __future__ import annotations

import numpy as np


class Layer:
    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.cache: tuple = ()

    def zero_grad(self) -> None:
        for k, v in self.params.items():
            self.grads[k] = np.zeros_like(v)


class Linear(Layer):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, dtype=np.float32, scale: float = 1.0):
        super().__init__()
        # He-style init for SiLU nets; ``scale`` shrinks residual outputs at init.
        self.params = {"W": (rng.standard_normal((n_in, n_out)) * scale * np.sqrt(2.0 / n_in)).astype(dtype),
                       "b": np.zeros(n_out, dtype=dtype)}
        self.zero_grad()

    def forward(self, x):
        self.cache = (x,)
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dy):
        (x,) = self.cache
        self.grads["W"] += x.T @ dy
        self.grads["b"] += dy.sum(axis=0)
        return dy @ self.params["W"].T


class SiLU(Layer):
    def forward(self, x):
        # exp of -|x| never overflows; the two branches are the same function.
        e = np.exp(-np.abs(x))
        s = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        self.cache = (x, s)
        return x * s

    def backward(self, dy):
        x, s = self.cache
        # d/dx [x s(x)] = s + x s (1 - s)
        return dy * (s * (1.0 + x * (1.0 - s)))


class LayerNorm(Layer):
    def __init__(self, n: int, dtype=np.float32, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.params = {"g": np.ones(n, dtype=dtype), "b": np.zeros(n, dtype=dtype)}
        self.zero_grad()

    def forward(self, x):
        mu = x.mean(axis=1, keepdims=True)
        inv = 1.0 / np.sqrt(x.var(axis=1, keepdims=True) + self.eps)
        xhat = (x - mu) * inv
        self.cache = (xhat, inv)
        return xhat * self.params["g"] + self.params["b"]

    def backward(self, dy):
        xhat, inv = self.cache
        n = xhat.shape[1]
        self.grads["g"] += (dy * xhat).sum(axis=0)
        self.grads["b"] += dy.sum(axis=0)
        dxhat = dy * self.params["g"]
        # Standard result: the mean and variance both depend on every input.
        return inv / n * (n * dxhat - dxhat.sum(axis=1, keepdims=True)
                          - xhat * (dxhat * xhat).sum(axis=1, keepdims=True))


class Embedding(Layer):
    def __init__(self, n: int, dim: int, rng: np.random.Generator, dtype=np.float32):
        super().__init__()
        self.params = {"E": (rng.standard_normal((n, dim)) * 0.02).astype(dtype)}
        self.zero_grad()

    def forward(self, idx):
        self.cache = (idx,)
        return self.params["E"][idx]

    def backward(self, dy):
        (idx,) = self.cache
        np.add.at(self.grads["E"], idx, dy)
        return None


def sinusoid(t: np.ndarray, dim: int) -> np.ndarray:
    """Transformer-style position encoding of the (integer) diffusion step."""
    half = dim // 2
    freqs = np.exp(-np.log(10_000.0) * np.arange(half) / half)
    a = t[:, None].astype(np.float64) * freqs[None, :]
    return np.concatenate([np.sin(a), np.cos(a)], axis=1)


class ResBlock(Layer):
    def __init__(self, width: int, rng, dtype=np.float32):
        super().__init__()
        self.norm = LayerNorm(width, dtype)
        self.a = Linear(width, width, rng, dtype)
        self.t = Linear(width, width, rng, dtype)
        self.act = SiLU()
        # Small output init: each block starts close to the identity.
        self.b = Linear(width, width, rng, dtype, scale=0.1)

    def layers(self):
        return {"norm": self.norm, "a": self.a, "t": self.t, "b": self.b}

    def forward(self, h, temb):
        u = self.a.forward(self.norm.forward(h)) + self.t.forward(temb)
        return h + self.b.forward(self.act.forward(u))

    def backward(self, dh):
        du = self.act.backward(self.b.backward(dh))
        dtemb = self.t.backward(du)
        return dh + self.norm.backward(self.a.backward(du)), dtemb


class Denoiser:
    """Predicts the noise in x_t. ``n_classes`` > 0 adds a class embedding,
    with index ``n_classes`` meaning "no class" (for classifier-free guidance)."""

    def __init__(self, dim: int = 784, width: int = 512, blocks: int = 3, temb_dim: int = 128,
                 n_classes: int = 0, skip: bool = True, seed: int = 0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.config = dict(dim=dim, width=width, blocks=blocks, temb_dim=temb_dim, n_classes=n_classes)
        self.dtype = dtype
        self.temb_dim = temb_dim
        self.t1 = Linear(temb_dim, width, rng, dtype)
        self.t_act = SiLU()
        self.t2 = Linear(width, width, rng, dtype)
        self.cls = Embedding(n_classes + 1, width, rng, dtype) if n_classes else None
        self.inp = Linear(dim, width, rng, dtype)
        self.blocks = [ResBlock(width, rng, dtype) for _ in range(blocks)]
        self.out_norm = LayerNorm(width, dtype)
        self.out_act = SiLU()
        self.out = Linear(width, dim, rng, dtype, scale=0.1)
        self.skip = Linear(dim, dim, rng, dtype, scale=0.0) if skip else None
        self.config["skip"] = skip

    def layers(self) -> dict[str, Layer]:
        out = {"t1": self.t1, "t2": self.t2, "inp": self.inp, "out_norm": self.out_norm, "out": self.out}
        if self.cls:
            out["cls"] = self.cls
        if self.skip:
            out["skip"] = self.skip
        for i, b in enumerate(self.blocks):
            out |= {f"block{i}.{k}": v for k, v in b.layers().items()}
        return out

    def parameters(self) -> dict[str, np.ndarray]:
        return {f"{n}.{k}": v for n, layer in self.layers().items() for k, v in layer.params.items()}

    def gradients(self) -> dict[str, np.ndarray]:
        return {f"{n}.{k}": v for n, layer in self.layers().items() for k, v in layer.grads.items()}

    def zero_grad(self) -> None:
        for layer in self.layers().values():
            layer.zero_grad()

    def forward(self, x, t, y=None):
        temb = self.t2.forward(self.t_act.forward(self.t1.forward(sinusoid(t, self.temb_dim).astype(self.dtype))))
        if self.cls is not None:
            if y is None:
                y = np.full(len(x), self.config["n_classes"])
            temb = temb + self.cls.forward(y)
        h = self.inp.forward(x)
        for b in self.blocks:
            h = b.forward(h, temb)
        y_out = self.out.forward(self.out_act.forward(self.out_norm.forward(h)))
        return y_out + self.skip.forward(x) if self.skip is not None else y_out

    def backward(self, dout):
        if self.skip is not None:
            self.skip.backward(dout)
        dh = self.out_norm.backward(self.out_act.backward(self.out.backward(dout)))
        dtemb = 0.0
        for b in reversed(self.blocks):
            dh, dt = b.backward(dh)
            dtemb = dtemb + dt
        self.inp.backward(dh)
        if self.cls is not None:
            self.cls.backward(dtemb)
        self.t1.backward(self.t_act.backward(self.t2.backward(dtemb)))

    def state(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.parameters().items()}

    def load(self, state: dict[str, np.ndarray]) -> None:
        for k, v in self.parameters().items():
            v[...] = state[k]
