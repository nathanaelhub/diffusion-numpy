"""MNIST, verified and cached; and a dependency-free PNG writer for sample grids."""

from __future__ import annotations

import hashlib
import os
import struct
import urllib.request
import zlib
from pathlib import Path

import numpy as np

URLS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    "https://s3.amazonaws.com/img-datasets/mnist.npz",
]
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"  # pinned by Keras
DATA = Path(os.environ.get("DIFFUSION_DATA") or Path(__file__).resolve().parents[2] / "data")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> Path:
    """A verified mnist.npz; an interrupted or corrupt download never becomes the cache."""
    path = DATA / "mnist.npz"
    if path.exists() and _sha256(path) == SHA256:
        return path
    DATA.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    errors = []
    for url in URLS:
        try:
            urllib.request.urlretrieve(url, part)
            if _sha256(part) != SHA256:
                raise ValueError("checksum mismatch")
            os.replace(part, path)
            return path
        except Exception as e:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {e}")
        finally:
            part.unlink(missing_ok=True)
    raise RuntimeError("could not download MNIST: " + "; ".join(errors))


def load(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Images flattened to 784 floats in [-1, 1] (the diffusion data range), and labels."""
    with np.load(download()) as f:
        x, y = f[f"x_{split}"], f[f"y_{split}"]
    return (x.reshape(len(x), -1).astype(np.float32) / 127.5 - 1.0), y.astype(np.int64)


def grid(images: np.ndarray, cols: int = 10, pad: int = 2) -> np.ndarray:
    """(n, 784) in [-1, 1] -> one uint8 image with the digits tiled."""
    n = len(images)
    rows = -(-n // cols)
    out = np.full((rows * (28 + pad) + pad, cols * (28 + pad) + pad), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        tile = ((np.clip(im, -1, 1) + 1) * 127.5).round().astype(np.uint8).reshape(28, 28)
        y0, x0 = pad + r * (28 + pad), pad + c * (28 + pad)
        out[y0:y0 + 28, x0:x0 + 28] = 255 - tile  # dark digits on white
    return out


def write_png(path: str | Path, gray: np.ndarray, scale: int = 2) -> None:
    """Grayscale uint8 array -> PNG, nearest-neighbour upscaled."""
    img = np.repeat(np.repeat(gray, scale, axis=0), scale, axis=1)
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
