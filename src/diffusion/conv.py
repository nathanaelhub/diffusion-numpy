"""Convolutional layers for the U-Net, NCHW, with hand-derived backward passes.

Same contract as the layers in ``nn.py``: ``forward`` caches what
``backward`` needs, ``backward`` returns the input gradient and adds
parameter gradients into ``self.grads``.
"""

from __future__ import annotations

import numpy as np

from .nn import Layer


class Conv2d(Layer):
    """Stride-1 convolution by im2col: gather every k x k patch into a row,
    then one matrix multiply.

    Inside the layer the data is channels-last (NHWC): a patch row is then k^2
    contiguous runs of channels, and the backward scatter (col2im) adds
    contiguous slices. Building patches from the NCHW window view with a
    transpose was 3x slower, all of it in strided copies.
    """

    def __init__(self, c_in: int, c_out: int, k: int, rng: np.random.Generator, padding: int | None = None,
                 dtype=np.float32, scale: float = 1.0):
        super().__init__()
        self.k, self.c_in, self.c_out = k, c_in, c_out
        self.pad = k // 2 if padding is None else padding
        fan_in = c_in * k * k
        # Rows are ordered (ky, kx, c_in), matching the patch layout below.
        self.params = {"W": (rng.standard_normal((fan_in, c_out)) * scale * np.sqrt(2.0 / fan_in)).astype(dtype),
                       "b": np.zeros(c_out, dtype=dtype)}
        self.zero_grad()

    def forward(self, x):
        n, c, h, w = x.shape
        p, k = self.pad, self.k
        xh = x.transpose(0, 2, 3, 1)
        if p:
            xh = np.pad(xh, ((0, 0), (p, p), (p, p), (0, 0)))
        ho, wo = xh.shape[1] - k + 1, xh.shape[2] - k + 1
        if k == 1:
            cols = np.ascontiguousarray(xh).reshape(-1, c)
        else:
            cols = np.concatenate([xh[:, i:i + ho, j:j + wo, :] for i in range(k) for j in range(k)],
                                  axis=3).reshape(-1, k * k * c)
        self.cache = (cols, x.shape, xh.shape, ho, wo)
        y = cols @ self.params["W"] + self.params["b"]
        return y.reshape(n, ho, wo, self.c_out).transpose(0, 3, 1, 2)

    def backward(self, dy):
        cols, xshape, pshape, ho, wo = self.cache
        n, c, h, w = xshape
        k, p = self.k, self.pad
        d2 = dy.transpose(0, 2, 3, 1).reshape(-1, self.c_out)
        self.grads["W"] += cols.T @ d2
        self.grads["b"] += d2.sum(axis=0)
        dcols = (d2 @ self.params["W"].T).reshape(n, ho, wo, k * k, c)
        if k == 1:
            dxh = dcols[:, :, :, 0, :]
        else:
            dxh = np.zeros(pshape, dtype=dy.dtype)
            for i in range(k):
                for j in range(k):
                    dxh[:, i:i + ho, j:j + wo, :] += dcols[:, :, :, i * k + j, :]
        if p:
            dxh = dxh[:, p:p + h, p:p + w, :]
        return dxh.transpose(0, 3, 1, 2)


class GroupNorm(Layer):
    """Normalize over (channels in the group, H, W) per sample: LayerNorm's
    derivation applied to each group, with a gain and bias per channel."""

    def __init__(self, groups: int, channels: int, dtype=np.float32, eps: float = 1e-5):
        super().__init__()
        if channels % groups:
            raise ValueError(f"{channels} channels don't split into {groups} groups")
        self.g, self.eps = groups, eps
        self.params = {"g": np.ones(channels, dtype=dtype), "b": np.zeros(channels, dtype=dtype)}
        self.zero_grad()

    def forward(self, x):
        n, c, h, w = x.shape
        xg = x.reshape(n, self.g, -1)
        mu = xg.mean(axis=2, keepdims=True)
        inv = 1.0 / np.sqrt(xg.var(axis=2, keepdims=True) + self.eps)
        xhat = ((xg - mu) * inv).reshape(x.shape)
        self.cache = (xhat, inv)
        return xhat * self.params["g"][None, :, None, None] + self.params["b"][None, :, None, None]

    def backward(self, dy):
        xhat, inv = self.cache
        n, c, h, w = xhat.shape
        self.grads["g"] += (dy * xhat).sum(axis=(0, 2, 3))
        self.grads["b"] += dy.sum(axis=(0, 2, 3))
        dxh = (dy * self.params["g"][None, :, None, None]).reshape(n, self.g, -1)
        xh = xhat.reshape(n, self.g, -1)
        m = xh.shape[2]
        dx = inv / m * (m * dxh - dxh.sum(axis=2, keepdims=True) - xh * (dxh * xh).sum(axis=2, keepdims=True))
        return dx.reshape(xhat.shape)


class AvgPool2(Layer):
    def forward(self, x):
        n, c, h, w = x.shape
        return x.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))

    def backward(self, dy):
        return np.repeat(np.repeat(dy, 2, axis=2), 2, axis=3) / 4.0


class Upsample2(Layer):
    """Nearest-neighbour x2."""

    def forward(self, x):
        return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)

    def backward(self, dy):
        n, c, h, w = dy.shape
        return dy.reshape(n, c, h // 2, 2, w // 2, 2).sum(axis=(3, 5))
