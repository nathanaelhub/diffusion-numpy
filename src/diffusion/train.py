"""Training loop: noise a batch at random steps, predict eps or v, MSE.

Checkpoints hold the weights, their EMA, the Adam state and the RNG state,
so an interrupted run resumes exactly where it stopped.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .nn import Denoiser
from .optim import EMA, Adam, clip_grad_norm
from .schedule import Schedule, target


def lr_at(step: int, steps: int, lr: float, warmup: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def save(path: Path, net: Denoiser, ema: EMA, opt: Adam, step: int, rng, meta: dict) -> None:
    arrays = {f"p/{k}": v for k, v in net.parameters().items()}
    arrays |= {f"ema/{k}": v for k, v in ema.shadow.items()}
    arrays |= {f"m/{k}": v for k, v in opt.m.items()} | {f"v/{k}": v for k, v in opt.v.items()}
    info = {"config": net.config, "step": step, "adam_t": opt.t, "rng": rng.bit_generator.state, **meta}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, info=np.array(json.dumps(info)), **arrays)
    tmp.replace(path)


def load(path: Path):
    """(net with EMA weights loaded, info) for sampling."""
    f = np.load(path)
    info = json.loads(str(f["info"]))
    net = Denoiser(**info["config"])
    net.load({k: f[f"ema/{k}"] for k in net.parameters()})
    return net, info


def train(out: Path, steps: int = 30_000, batch: int = 128, lr: float = 1e-3, warmup: int = 500,
          width: int = 512, blocks: int = 3, n_classes: int = 10, skip: bool = True, p_uncond: float = 0.1,
          schedule: str = "cosine", pred: str = "v", grad_clip: float = 1.0, ema_decay: float = 0.999,
          every: int = 1000, seed: int = 0, resume: bool = False, stop_at: int | None = None,
          log=print) -> Path:
    from .data import load as load_mnist

    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "model.npz"
    x_all, y_all = load_mnist("train")
    sched = Schedule(kind=schedule)
    net = Denoiser(width=width, blocks=blocks, n_classes=n_classes, skip=skip, seed=seed)
    opt = Adam(net.parameters(), lr=lr)
    ema = EMA(net.parameters(), ema_decay)
    rng = np.random.default_rng(seed)
    start = 0
    meta = {"pred": pred, "schedule": schedule, "p_uncond": p_uncond, "batch": batch, "steps": steps, "lr": lr}
    if resume and ckpt.exists():
        f = np.load(ckpt)
        info = json.loads(str(f["info"]))
        net.load({k: f[f"p/{k}"] for k in net.parameters()})
        for k in ema.shadow:
            ema.shadow[k][...] = f[f"ema/{k}"]
            opt.m[k][...] = f[f"m/{k}"]
            opt.v[k][...] = f[f"v/{k}"]
        opt.t, start = info["adam_t"], info["step"]
        rng.bit_generator.state = info["rng"]
        log(f"resumed at step {start}")

    log_path = out / "loss.csv"
    if start == 0:
        log_path.write_text("step,loss,grad_norm,lr\n")
    running, t0 = None, time.perf_counter()
    # ``stop_at`` ends a run early on the same schedule, as an interruption would.
    for step in range(start, min(steps, stop_at or steps)):
        idx = rng.integers(0, len(x_all), batch)
        x0 = x_all[idx]
        t = rng.integers(0, sched.T, batch)
        eps = rng.standard_normal(x0.shape).astype(np.float32)
        xt = sched.noise(x0, t, eps).astype(np.float32)
        y = None
        if n_classes:
            # Drop the label some of the time so one network learns both the
            # conditional and the unconditional model (classifier-free guidance).
            y = np.where(rng.random(batch) < p_uncond, n_classes, y_all[idx])

        net.zero_grad()
        diff = net.forward(xt, t, y) - target(x0, eps, sched.abar[t], pred)
        loss = float((diff ** 2).mean())
        net.backward(2.0 * diff / diff.size)
        grads = net.gradients()
        gnorm = clip_grad_norm(grads, grad_clip)
        rate = lr_at(step, steps, lr, warmup)
        opt.step(grads, rate)
        ema.update(net.parameters())

        running = loss if running is None else 0.99 * running + 0.01 * loss
        if (step + 1) % 100 == 0:
            with log_path.open("a") as fh:
                fh.write(f"{step + 1},{running:.5f},{gnorm:.4f},{rate:.3e}\n")
        if (step + 1) % every == 0 or step + 1 == steps or step + 1 == stop_at:
            save(ckpt, net, ema, opt, step + 1, rng, meta)
            log(f"step {step + 1}/{steps}  loss {running:.4f}  {time.perf_counter() - t0:.0f}s")
    return ckpt
