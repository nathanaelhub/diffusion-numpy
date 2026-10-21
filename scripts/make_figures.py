"""Figures for the README, from a trained model.

    uv run python scripts/make_figures.py runs/main/model.npz
    uv run python scripts/make_figures.py runs/unet/model.npz unet- 0

The optional arguments are a prefix for the file names and the step to stop
sampling at (400 for the MLP, see the README).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from diffusion.data import grid, write_png
from diffusion.schedule import Schedule, _predict, ddim_sample, eps_and_x0
from diffusion.train import load

DOCS = Path(__file__).resolve().parents[1] / "docs"


def trajectory(net, sched, pred, digits, steps=50, show=10, seed=0, guidance=2.0):
    """The model's running estimate of the clean image (x0) along one DDIM run."""
    rng = np.random.default_rng(seed)
    ts = np.linspace(sched.T - 1, 0, steps).round().astype(int)
    keep = set(np.linspace(0, steps - 1, show).round().astype(int))
    x = rng.standard_normal((len(digits), 784)).astype(net.dtype)
    frames = []
    for i, t in enumerate(ts):
        ab = sched.abar[t]
        _, x0 = eps_and_x0(_predict(net, x, t, digits, guidance), x, ab, pred)
        if i in keep:
            frames.append(x0.copy())
        if i == len(ts) - 1:
            break
        ab_next = sched.abar[ts[i + 1]]
        eps = (x - np.sqrt(ab) * x0) / np.sqrt(1 - ab)
        x = (np.sqrt(ab_next) * x0 + np.sqrt(1 - ab_next) * eps).astype(net.dtype)
    # rows = digits, columns = time
    return np.stack(frames, axis=1).reshape(-1, 784)


def main(model: str, prefix: str = "", t_min: int = 400) -> None:
    net, info = load(Path(model))
    sched, pred = Schedule(kind=info["schedule"]), info["pred"]
    DOCS.mkdir(exist_ok=True)
    per_class = np.repeat(np.arange(10), 10)

    # For the MLP, the settings with the lowest Frechet distance in results/ (w = 2, stop at t = 400).
    x = ddim_sample(net, sched, 100, np.random.default_rng(0), y=per_class, guidance=2.0, pred=pred, t_min=t_min)
    write_png(DOCS / f"{prefix}samples.png", grid(x))
    if t_min:
        x = ddim_sample(net, sched, 100, np.random.default_rng(0), y=per_class, guidance=2.0, pred=pred)
        write_png(DOCS / f"{prefix}samples-full.png", grid(x))
    x = ddim_sample(net, sched, 100, np.random.default_rng(0), pred=pred, t_min=t_min)
    write_png(DOCS / f"{prefix}unconditional.png", grid(x))

    # Same starting noise, increasing guidance: columns w = 1, 1.5, 2, 3, 5, 8.
    ws = [1, 1.5, 2, 3, 5, 8]
    digits = np.array([2, 3, 5, 8, 9])
    cols = [ddim_sample(net, sched, len(digits), np.random.default_rng(7), y=digits, guidance=w, pred=pred,
                        t_min=t_min) for w in ws]
    write_png(DOCS / f"{prefix}guidance-sweep.png", grid(np.stack(cols, axis=1).reshape(-1, 784), cols=len(ws)))

    write_png(DOCS / f"{prefix}trajectory.png", grid(trajectory(net, sched, pred, np.arange(10)), cols=10))
    print("wrote", ", ".join(sorted(p.name for p in DOCS.glob(f"{prefix}*.png"))))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "runs/main/model.npz", *args[1:2], *(int(a) for a in args[2:3]))
