"""axono.fused 融合算子测试: 数值 vs 手工分解实现。"""


import numpy as np
import pytest

import axono
from axono import Tensor


def _rsilu(x):
    return x / (1.0 + np.exp(-x))


def _t(arr, device="cpu"):
    t = Tensor.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return t.to(device) if device != "cpu" else t


class TestSiluMul:
    def test_matches_reference_cpu(self):
        rng = np.random.RandomState(0)
        g = rng.randn(37, 53).astype(np.float32)
        u = rng.randn(37, 53).astype(np.float32)
        y = axono.fused.silu_mul(_t(g), _t(u)).to_numpy()
        ref = _rsilu(g) * u
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-6)

    def test_matches_reference_cuda(self, cuda_env):
        rng = np.random.RandomState(0)
        g = rng.randn(37, 53).astype(np.float32)
        u = rng.randn(37, 53).astype(np.float32)
        y = axono.fused.silu_mul(_t(g, "cuda"), _t(u, "cuda")).to_numpy()
        np.testing.assert_allclose(y, _rsilu(g) * u, rtol=1e-5, atol=1e-6)

    def test_negative_values(self):
        g = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
        u = np.ones(5, dtype=np.float32)
        y = axono.fused.silu_mul(_t(g), _t(u)).to_numpy()
        ref = _rsilu(g)
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-6)


class TestAddRmsNorm:
    def test_matches_reference_cpu(self):
        rng = np.random.RandomState(1)
        x = rng.randn(4, 128).astype(np.float32)
        r = rng.randn(4, 128).astype(np.float32)
        w = rng.randn(128).astype(np.float32)
        eps = 1e-6
        yt, out = axono.fused.add_rms_norm(
            _t(x), _t(r), _t(w), eps
        )
        y = x + r
        np.testing.assert_allclose(yt.to_numpy(), y, rtol=1e-5, atol=1e-6)
        ref = y / np.sqrt((y**2).mean(-1, keepdims=True) + eps) * w
        np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-4, atol=1e-5)

    def test_matches_reference_cuda(self, cuda_env):
        rng = np.random.RandomState(1)
        x = rng.randn(4, 128).astype(np.float32)
        r = rng.randn(4, 128).astype(np.float32)
        w = rng.randn(128).astype(np.float32)
        eps = 1e-6
        yt, out = axono.fused.add_rms_norm(
            _t(x, "cuda"), _t(r, "cuda"), _t(w, "cuda"), eps
        )
        y = x + r
        np.testing.assert_allclose(yt.to_numpy(), y, rtol=1e-5, atol=1e-6)
        ref = y / np.sqrt((y**2).mean(-1, keepdims=True) + eps) * w
        np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-4, atol=1e-5)

    def test_y_feeds_next_layer(self):
        """y (残差和) 与下一次 add_rms_norm 的 residual 语义一致。"""
        rng = np.random.RandomState(2)
        h = rng.randn(2, 64).astype(np.float32)
        a1 = rng.randn(2, 64).astype(np.float32)
        a2 = rng.randn(2, 64).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        h1, _ = axono.fused.add_rms_norm(_t(a1), _t(h), _t(w), 1e-6)
        h2, _ = axono.fused.add_rms_norm(_t(a2), h1, _t(w), 1e-6)
        np.testing.assert_allclose(
            h2.to_numpy(), a2 + a1 + h, rtol=1e-5, atol=1e-6
        )
