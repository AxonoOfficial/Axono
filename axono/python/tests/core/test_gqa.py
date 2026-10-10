"""axono.gqa_decode_attention 测试: 数值 vs SDPA 与 numpy 参考。"""

import math

import numpy as np

import axono
from axono import Tensor


def _ref_gqa(q, k, v):
    """q (1,hq,d), k/v (skv,hkv,d) → (1,hq,d)"""
    sq, hq, d = q.shape
    skv, hkv = k.shape[:2]
    group = hq // hkv
    scale = 1.0 / math.sqrt(d)
    out = np.zeros((1, hq, d), np.float32)
    for h in range(hq):
        hk = h // group
        scores = (q[0, h] @ k[:, hk].T) * scale
        scores = scores - scores.max()
        w = np.exp(scores)
        w /= w.sum()
        out[0, h] = w @ v[:, hk]
    return out


def _t(arr, device="cpu"):
    t = Tensor.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return t.to(device) if device != "cpu" else t


def test_gqa_decode_matches_sdpa(cuda_env):
    rng = np.random.RandomState(3)
    for skv in (1, 7, 256, 512, 1024):
        q = rng.randn(1, 16, 128).astype(np.float32)
        k = rng.randn(skv, 8, 128).astype(np.float32)
        v = rng.randn(skv, 8, 128).astype(np.float32)
        qt, kt, vt = (_t(a, "cuda") for a in (q, k, v))
        out = axono.gqa_decode_attention(qt, kt, vt).to_numpy()
        ref = _ref_gqa(q, k, v)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_gqa_decode_matches_sdpa_op(cuda_env):
    """与通用 SDPA (非因果) 输出一致。"""
    rng = np.random.RandomState(4)
    skv = 300
    q = rng.randn(1, 16, 128).astype(np.float32)
    k = rng.randn(skv, 8, 128).astype(np.float32)
    v = rng.randn(skv, 8, 128).astype(np.float32)
    qt, kt, vt = (_t(a, "cuda") for a in (q, k, v))
    a1 = axono.gqa_decode_attention(qt, kt, vt).to_numpy()
    a2 = axono.scaled_dot_product_attention(qt, kt, vt, False).to_numpy()
    np.testing.assert_allclose(a1, a2, rtol=1e-4, atol=1e-4)
