"""Judging samples with a classifier that never saw them.

There is no Inception network in a NumPy-only project, so an MNIST classifier
(an MLP, also hand-written here) stands in for it:

- **confidence**: mean top-class probability; scribbles get low confidence;
- **class coverage**: how evenly the 10 digits appear (mode collapse shows
  up as a skewed histogram);
- **conditional accuracy**: for class-conditional samples, how often the
  classifier names the digit that was asked for;
- **Fréchet distance** between classifier features of samples and of real
  test images (the FID recipe on a smaller feature extractor). Only
  comparable within this project, not to published FID numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .nn import Linear, SiLU
from .optim import Adam


class Classifier:
    def __init__(self, seed: int = 0, hidden: int = 256):
        rng = np.random.default_rng(seed)
        self.l1, self.a1 = Linear(784, hidden, rng), SiLU()
        self.l2, self.a2 = Linear(hidden, hidden, rng), SiLU()
        self.l3 = Linear(hidden, 10, rng)
        self.layers = {"l1": self.l1, "l2": self.l2, "l3": self.l3}

    def params(self):
        return {f"{n}.{k}": v for n, l in self.layers.items() for k, v in l.params.items()}

    def features(self, x):
        return self.a2.forward(self.l2.forward(self.a1.forward(self.l1.forward(x))))

    def logits(self, x):
        return self.l3.forward(self.features(x))

    def probs(self, x, chunk: int = 2000):
        out = []
        for i in range(0, len(x), chunk):
            z = self.logits(x[i:i + chunk])
            z = np.exp(z - z.max(axis=1, keepdims=True))
            out.append(z / z.sum(axis=1, keepdims=True))
        return np.concatenate(out)

    def train(self, x, y, epochs: int = 8, batch: int = 128, lr: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        opt = Adam(self.params(), lr=lr)
        for _ in range(epochs):
            for i in np.array_split(rng.permutation(len(x)), len(x) // batch):
                for l in self.layers.values():
                    l.zero_grad()
                z = self.logits(x[i])
                p = np.exp(z - z.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                p[np.arange(len(i)), y[i]] -= 1  # softmax + cross-entropy gradient
                d = self.l3.backward(p / len(i))
                self.l1.backward(self.a1.backward(self.l2.backward(self.a2.backward(d))))
                opt.step({f"{n}.{k}": g for n, l in self.layers.items() for k, g in l.grads.items()})
        return self

    def save(self, path: Path):
        np.savez(path, **self.params())

    def load(self, path: Path):
        f = np.load(path)
        for k, v in self.params().items():
            v[...] = f[k]
        return self


def classifier(cache: Path) -> Classifier:
    from .data import load

    if cache.exists():
        return Classifier().load(cache)
    x, y = load("train")
    clf = Classifier().train(x, y)
    cache.parent.mkdir(parents=True, exist_ok=True)
    clf.save(cache)
    return clf


def _sqrt_trace(a: np.ndarray, b: np.ndarray) -> float:
    """Tr sqrt(A B) for symmetric PSD A, B, via the symmetric A^1/2 B A^1/2."""
    w, v = np.linalg.eigh(a)
    ra = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = ra @ b @ ra
    return float(np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum())


def frechet(f1: np.ndarray, f2: np.ndarray) -> float:
    f1, f2 = f1.astype(np.float64), f2.astype(np.float64)
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * _sqrt_trace(s1, s2))


def score(clf: Classifier, samples: np.ndarray, reference: np.ndarray, asked: np.ndarray | None = None) -> dict:
    p = clf.probs(samples)
    pred = p.argmax(1)
    hist = np.bincount(pred, minlength=10) / len(pred)
    out = {
        "n": len(samples),
        "confidence": float(p.max(1).mean()),
        "class_share": [round(float(h), 3) for h in hist],
        # 1.0 = all ten digits equally often; lower = some digits missing.
        "coverage": float(np.exp(-(hist[hist > 0] * np.log(hist[hist > 0])).sum()) / 10),
        "frechet": frechet(clf.features(samples), clf.features(reference)),
    }
    if asked is not None:
        out["conditional_accuracy"] = float((pred == asked).mean())
    return out
