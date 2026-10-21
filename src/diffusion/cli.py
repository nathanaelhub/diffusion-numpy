"""ddpm train | sample | evaluate"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ddpm", description="Denoising diffusion from scratch in NumPy, on MNIST.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--out", type=Path, default=Path("runs/default"))
    t.add_argument("--steps", type=int, default=30_000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--arch", choices=("mlp", "unet"), default="mlp")
    t.add_argument("--channels", default="16,32", help="U-Net channels at 28x28 and 14x14/7x7")
    t.add_argument("--width", type=int, default=512)
    t.add_argument("--blocks", type=int, default=3)
    t.add_argument("--unconditional", action="store_true", help="no class labels (no guidance)")
    t.add_argument("--schedule", choices=("cosine", "linear"), default="cosine")
    t.add_argument("--pred", choices=("v", "eps"), default="v", help="what the network predicts")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action="store_true")

    s = sub.add_parser("sample")
    s.add_argument("--model", type=Path, default=Path("runs/default/model.npz"))
    s.add_argument("--out", type=Path, default=Path("samples.png"))
    s.add_argument("-n", type=int, default=100)
    s.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim")
    s.add_argument("--t-min", type=int, default=0, help="DDIM: stop at this step and return the x0 estimate")
    s.add_argument("--steps", type=int, default=50, help="DDIM steps")
    s.add_argument("--digit", type=int, help="generate this digit (class-conditional models)")
    s.add_argument("--per-class", action="store_true", help="one row per digit 0-9")
    s.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance weight")
    s.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("evaluate")
    e.add_argument("--model", type=Path, default=Path("runs/default/model.npz"))
    e.add_argument("-n", type=int, default=2000)
    e.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim")
    e.add_argument("--t-min", type=int, default=0, help="DDIM: stop at this step and return the x0 estimate")
    e.add_argument("--steps", type=int, default=50)
    e.add_argument("--guidance", type=float, default=1.0)
    e.add_argument("--conditional", action="store_true", help="ask for each digit equally and score accuracy")
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--baselines", action="store_true", help="also score real images and noise")

    a = ap.parse_args(argv)
    if a.cmd == "train":
        from .train import train
        train(a.out, steps=a.steps, batch=a.batch, lr=a.lr, width=a.width, blocks=a.blocks, arch=a.arch,
              channels=tuple(int(c) for c in a.channels.split(",")),
              n_classes=0 if a.unconditional else 10, schedule=a.schedule, pred=a.pred, seed=a.seed, resume=a.resume)
        return 0

    from .data import grid, load, write_png
    from .schedule import Schedule, ddim_sample, ddpm_sample
    from .train import load as load_model

    net, info = load_model(a.model)
    sched = Schedule(kind=info.get("schedule", "cosine"))
    rng = np.random.default_rng(a.seed)

    def generate(n, y):
        pred = info.get("pred", "eps")
        if a.sampler == "ddpm":
            return ddpm_sample(net, sched, n, rng, y=y, guidance=a.guidance, pred=pred)
        return ddim_sample(net, sched, n, rng, steps=a.steps, y=y, guidance=a.guidance, pred=pred, t_min=a.t_min)

    if a.cmd == "sample":
        y = None
        if a.per_class:
            y = np.repeat(np.arange(10), max(1, a.n // 10))
        elif a.digit is not None:
            y = np.full(a.n, a.digit)
        t0 = time.perf_counter()
        x = generate(len(y) if y is not None else a.n, y)
        write_png(a.out, grid(x, cols=10))
        print(f"wrote {a.out} ({len(x)} samples, {time.perf_counter() - t0:.1f}s)")
        return 0

    from .evaluate import classifier, score

    clf = classifier(Path("runs/classifier.npz"))
    x_test, y_test = load("test")
    ref = x_test[: a.n]
    asked = np.arange(a.n) % 10 if a.conditional else None
    # Chunks bound memory; the U-Net's im2col buffers need smaller ones.
    chunk = 32 if net.config.get("arch") == "unet" else 500
    t0 = time.perf_counter()
    x = np.concatenate([generate(len(c), c if asked is not None else None)
                        for c in np.array_split(asked if asked is not None else np.zeros(a.n, int), max(1, a.n // chunk))])
    report = {"model": str(a.model), "step": info["step"], "sampler": a.sampler,
              "steps": a.steps if a.sampler == "ddim" else sched.T, "guidance": a.guidance,
              "t_min": a.t_min if a.sampler == "ddim" else 0,
              "seconds": round(time.perf_counter() - t0, 1), "samples": score(clf, x, ref, asked)}
    if a.baselines:
        x_train, y_train = load("train")
        report["real_train_images"] = score(clf, x_train[: a.n], ref, y_train[: a.n])
        report["gaussian_noise"] = score(clf, np.clip(rng.standard_normal((a.n, 784)), -1, 1).astype(np.float32), ref)
        report["classifier_test_accuracy"] = float((clf.probs(x_test).argmax(1) == y_test).mean())
    print(json.dumps(report, indent=2))
    return 0
