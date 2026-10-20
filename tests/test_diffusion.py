import zlib

import numpy as np
import pytest

from diffusion import data, evaluate, gradcheck, train
from diffusion.nn import Denoiser, LayerNorm, sinusoid
from diffusion.schedule import Schedule, ddim_sample, ddpm_sample


@pytest.mark.parametrize("n_classes", [0, 10])
def test_backward_matches_finite_differences(n_classes):
    for seed in range(3):
        assert gradcheck.check(n_classes=n_classes, seed=seed) <= 1.0


def test_layernorm_output_is_normalized():
    x = np.random.default_rng(0).standard_normal((4, 32)) * 5 + 3
    y = LayerNorm(32, dtype=np.float64).forward(x)
    assert np.allclose(y.mean(1), 0, atol=1e-7) and np.allclose(y.std(1), 1, atol=1e-3)


def test_sinusoid_distinguishes_steps():
    e = sinusoid(np.array([0, 1, 999]), 16)
    assert e.shape == (3, 16) and not np.allclose(e[0], e[1])


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_schedule_destroys_signal_monotonically(kind):
    s = Schedule(kind=kind)
    assert np.all(np.diff(s.abar) < 0)
    assert s.abar[0] > 0.99 and s.abar[-1] < 1e-3


class Oracle:
    """Knows the clean image, so it returns the exact eps (or v) for x_t."""

    def __init__(self, x0, sched, kind="eps"):
        self.x0, self.sched, self.kind = x0, sched, kind
        self.dtype, self.config = np.float64, {"n_classes": 0}

    def forward(self, x, t, y=None):
        ab = self.sched.abar[t][:, None]
        eps = (x - np.sqrt(ab) * self.x0) / np.sqrt(1 - ab)
        return eps if self.kind == "eps" else np.sqrt(ab) * eps - np.sqrt(1 - ab) * self.x0


@pytest.mark.parametrize("kind", ["eps", "v"])
def test_samplers_recover_the_image_with_perfect_predictions(kind):
    sched = Schedule()
    rng = np.random.default_rng(0)
    x0 = np.clip(rng.standard_normal((3, 784)) * 0.5, -1, 1)
    net = Oracle(x0, sched, kind)
    out = ddim_sample(net, sched, 3, np.random.default_rng(1), steps=20, pred=kind)
    assert np.allclose(out, x0, atol=1e-6)
    assert np.allclose(ddpm_sample(net, sched, 3, np.random.default_rng(1), pred=kind), x0, atol=1e-6)


def test_v_target_round_trips():
    from diffusion.schedule import eps_and_x0, target
    s = Schedule()
    rng = np.random.default_rng(0)
    x0, eps = rng.uniform(-1, 1, (4, 6)), rng.standard_normal((4, 6))
    for t in (0, 500, 999):
        tt = np.full(4, t)
        v = target(x0, eps, s.abar[tt], "v")
        e, x = eps_and_x0(v, s.noise(x0, tt, eps), s.abar[t], "v")
        assert np.allclose(e, eps) and np.allclose(x, x0)


def test_noising_matches_the_closed_form():
    s = Schedule()
    x0, eps = np.ones((2, 4)), np.zeros((2, 4))
    xt = s.noise(x0, np.array([0, 999]), eps)
    assert np.allclose(xt[:, 0], np.sqrt(s.abar[[0, 999]]))


def test_guidance_extrapolates_from_unconditional_to_conditional():
    from diffusion.schedule import _predict

    class Net:
        dtype, config = np.float32, {"n_classes": 10}

        def forward(self, x, t, y):
            return np.where(y[:, None] == 10, 0.0, 1.0) * np.ones_like(x)  # cond 1, uncond 0

    out = _predict(Net(), np.zeros((2, 3)), 5, np.array([3, 4]), guidance=3.0)
    assert np.allclose(out, 3.0)


def test_frechet_distance():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4000, 5))
    assert evaluate.frechet(a, a) == pytest.approx(0, abs=1e-6)
    # Same covariance, mean shifted by 2 in every dimension: distance = 5 * 2^2.
    assert evaluate.frechet(a, a + 2) == pytest.approx(20, rel=1e-6)


def test_png_writer_produces_a_decodable_image(tmp_path):
    img = data.grid(np.linspace(-1, 1, 3 * 784).reshape(3, 784), cols=3)
    path = tmp_path / "g.png"
    data.write_png(path, img, scale=1)
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    idat = raw[raw.index(b"IDAT") + 4:raw.index(b"IEND") - 8]
    rows = zlib.decompress(idat)
    assert len(rows) == img.shape[0] * (img.shape[1] + 1)


def test_resume_reproduces_an_uninterrupted_run(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    fake = (rng.uniform(-1, 1, (64, 784)).astype(np.float32), rng.integers(0, 10, 64))
    monkeypatch.setattr(data, "load", lambda split="train": fake)
    kw = dict(steps=20, batch=8, width=32, blocks=1, every=10, log=lambda s: None)

    train.train(tmp_path / "straight", **kw)
    train.train(tmp_path / "split", **kw, stop_at=10)  # "interrupted" at step 10
    ck = np.load(tmp_path / "split" / "model.npz")
    import json
    info = json.loads(str(ck["info"]))
    assert info["step"] == 10
    train.train(tmp_path / "split", **kw, resume=True)

    a, b = np.load(tmp_path / "straight" / "model.npz"), np.load(tmp_path / "split" / "model.npz")
    for k in a.files:
        if k != "info":
            assert np.array_equal(a[k], b[k]), k


def test_denoiser_learns_on_a_tiny_problem():
    """Two fixed images: 300 steps must at least halve the untrained net's error."""
    from diffusion.optim import Adam
    rng = np.random.default_rng(0)
    imgs = rng.uniform(-1, 1, (2, 784)).astype(np.float32)
    s = Schedule()
    net = Denoiser(width=64, blocks=1, seed=0)
    t_ev = rng.integers(0, 1000, 256)
    x_ev = imgs[rng.integers(0, 2, 256)]
    e_ev = rng.standard_normal(x_ev.shape).astype(np.float32)
    xt_ev = s.noise(x_ev, t_ev, e_ev).astype(np.float32)

    def mse():
        return float(((net.forward(xt_ev, t_ev) - e_ev) ** 2).mean())

    before = mse()
    opt = Adam(net.parameters(), lr=3e-3)
    for _ in range(300):
        x0 = imgs[rng.integers(0, 2, 32)]
        t = rng.integers(0, 1000, 32)
        eps = rng.standard_normal(x0.shape).astype(np.float32)
        net.zero_grad()
        d = net.forward(s.noise(x0, t, eps).astype(np.float32), t) - eps
        net.backward(2 * d / d.size)
        opt.step(net.gradients())
    assert mse() < 0.5 * before
