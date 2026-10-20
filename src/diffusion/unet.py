"""A small convolutional U-Net denoiser with hand-derived gradients.

Same interface as ``nn.Denoiser``: images come in and go out flattened,
(batch, 784), so the samplers, training loop and evaluation are shared.

    28x28  in-conv -> Block(c1) ----------------------------- skip ---.
    14x14            pool -> Block(c1->c2) ------------- skip ---.     |
     7x7                     pool -> Block(c2) -> Block(c2)       |     |
    14x14                     upsample -> concat -> Block(2c2->c2)      |
    28x28                               upsample -> concat -> Block(c2+c1->c1)
                                                   -> GroupNorm -> SiLU -> out-conv

Each block adds a per-channel projection of the time (and class) embedding,
so conditioning reaches every resolution.
"""

from __future__ import annotations

import numpy as np

from .conv import AvgPool2, Conv2d, GroupNorm, Upsample2
from .nn import Embedding, Linear, SiLU, sinusoid


def _groups(c: int, want: int) -> int:
    g = min(want, c)
    while c % g:
        g -= 1
    return g


class Block:
    """GroupNorm -> SiLU -> conv3x3 -> + time projection -> GroupNorm -> SiLU
    -> conv3x3, plus a residual (a 1x1 conv when the channel count changes)."""

    def __init__(self, c_in: int, c_out: int, temb: int, groups: int, rng, dtype):
        self.n1, self.a1 = GroupNorm(_groups(c_in, groups), c_in, dtype), SiLU()
        self.c1 = Conv2d(c_in, c_out, 3, rng, dtype=dtype)
        self.t = Linear(temb, c_out, rng, dtype)
        self.n2, self.a2 = GroupNorm(_groups(c_out, groups), c_out, dtype), SiLU()
        # Small init on the second conv: each block starts near the identity.
        self.c2 = Conv2d(c_out, c_out, 3, rng, dtype=dtype, scale=0.1)
        self.skip = Conv2d(c_in, c_out, 1, rng, dtype=dtype) if c_in != c_out else None

    def layers(self):
        out = {"n1": self.n1, "c1": self.c1, "t": self.t, "n2": self.n2, "c2": self.c2}
        if self.skip:
            out["skip"] = self.skip
        return out

    def forward(self, x, temb):
        h = self.c1.forward(self.a1.forward(self.n1.forward(x))) + self.t.forward(temb)[:, :, None, None]
        h = self.c2.forward(self.a2.forward(self.n2.forward(h)))
        return (self.skip.forward(x) if self.skip else x) + h

    def backward(self, dy):
        dx = self.skip.backward(dy) if self.skip else dy
        dh = self.n2.backward(self.a2.backward(self.c2.backward(dy)))
        dtemb = self.t.backward(dh.sum(axis=(2, 3)))
        dx = dx + self.n1.backward(self.a1.backward(self.c1.backward(dh)))
        return dx, dtemb


class UNet:
    def __init__(self, dim: int = 784, channels: tuple[int, int] = (32, 64), groups: int = 8,
                 temb_dim: int = 128, temb_width: int = 128, n_classes: int = 0, seed: int = 0,
                 dtype=np.float32, **_ignored):
        side = int(round(dim ** 0.5))
        if side * side != dim or side % 4:
            raise ValueError(f"dim {dim} must be a square image with a side divisible by 4")
        c1, c2 = channels
        rng = np.random.default_rng(seed)
        self.side, self.dtype, self.temb_dim = side, dtype, temb_dim
        self.config = dict(arch="unet", dim=dim, channels=[c1, c2], groups=groups, temb_dim=temb_dim,
                           temb_width=temb_width, n_classes=n_classes)
        tw = temb_width
        self.t1, self.t_act, self.t2 = Linear(temb_dim, tw, rng, dtype), SiLU(), Linear(tw, tw, rng, dtype)
        self.cls = Embedding(n_classes + 1, tw, rng, dtype) if n_classes else None
        self.inp = Conv2d(1, c1, 3, rng, dtype=dtype)
        self.d1 = Block(c1, c1, tw, groups, rng, dtype)
        self.pool1 = AvgPool2()
        self.d2 = Block(c1, c2, tw, groups, rng, dtype)
        self.pool2 = AvgPool2()
        self.m1 = Block(c2, c2, tw, groups, rng, dtype)
        self.m2 = Block(c2, c2, tw, groups, rng, dtype)
        self.up1 = Upsample2()
        self.u1 = Block(2 * c2, c2, tw, groups, rng, dtype)
        self.up2 = Upsample2()
        self.u2 = Block(c2 + c1, c1, tw, groups, rng, dtype)
        self.out_norm, self.out_act = GroupNorm(_groups(c1, groups), c1, dtype), SiLU()
        self.out = Conv2d(c1, 1, 3, rng, dtype=dtype, scale=0.1)
        self.blocks = {"d1": self.d1, "d2": self.d2, "m1": self.m1, "m2": self.m2, "u1": self.u1, "u2": self.u2}

    def layers(self):
        out = {"t1": self.t1, "t2": self.t2, "inp": self.inp, "out_norm": self.out_norm, "out": self.out}
        if self.cls:
            out["cls"] = self.cls
        for name, b in self.blocks.items():
            out |= {f"{name}.{k}": v for k, v in b.layers().items()}
        return out

    def parameters(self):
        return {f"{n}.{k}": v for n, layer in self.layers().items() for k, v in layer.params.items()}

    def gradients(self):
        return {f"{n}.{k}": v for n, layer in self.layers().items() for k, v in layer.grads.items()}

    def zero_grad(self):
        for layer in self.layers().values():
            layer.zero_grad()

    def state(self):
        return {k: v.copy() for k, v in self.parameters().items()}

    def load(self, state):
        for k, v in self.parameters().items():
            v[...] = state[k]

    def forward(self, x, t, y=None):
        temb = self.t2.forward(self.t_act.forward(self.t1.forward(sinusoid(t, self.temb_dim).astype(self.dtype))))
        if self.cls is not None:
            if y is None:
                y = np.full(len(x), self.config["n_classes"])
            temb = temb + self.cls.forward(y)
        s = self.side
        h1 = self.d1.forward(self.inp.forward(x.reshape(len(x), 1, s, s)), temb)
        h2 = self.d2.forward(self.pool1.forward(h1), temb)
        m = self.m2.forward(self.m1.forward(self.pool2.forward(h2), temb), temb)
        u = self.u1.forward(np.concatenate([self.up1.forward(m), h2], axis=1), temb)
        u = self.u2.forward(np.concatenate([self.up2.forward(u), h1], axis=1), temb)
        out = self.out.forward(self.out_act.forward(self.out_norm.forward(u)))
        return out.reshape(len(x), -1)

    def backward(self, dout):
        c2 = self.config["channels"][1]
        n = len(dout)
        du = self.out_norm.backward(self.out_act.backward(self.out.backward(dout.reshape(n, 1, self.side, self.side))))
        dtemb = 0.0

        dcat, dt = self.u2.backward(du)
        dtemb = dtemb + dt
        dh1 = dcat[:, c2:]  # skip from d1
        du, dt = self.u1.backward(self.up2.backward(dcat[:, :c2]))
        dtemb = dtemb + dt
        dh2 = du[:, c2:]  # skip from d2
        dm, dt = self.m2.backward(self.up1.backward(du[:, :c2]))
        dtemb = dtemb + dt
        dm, dt = self.m1.backward(dm)
        dtemb = dtemb + dt
        dh2 = dh2 + self.pool2.backward(dm)
        dp1, dt = self.d2.backward(dh2)
        dtemb = dtemb + dt
        dh1 = dh1 + self.pool1.backward(dp1)
        dx, dt = self.d1.backward(dh1)
        dtemb = dtemb + dt
        self.inp.backward(dx)

        if self.cls is not None:
            self.cls.backward(dtemb)
        self.t1.backward(self.t_act.backward(self.t2.backward(dtemb)))
