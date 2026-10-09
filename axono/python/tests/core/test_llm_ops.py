# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLM 相关算子与单算子融合 Linear 测试 (CPU/CUDA)。"""

import math

import numpy as np
import pytest

import axono
from axono import DataType, Tensor

np.random.seed(2026)


def _to_numpy(t: Tensor) -> np.ndarray:
    return t.to_numpy()


def _randn(shape, device, dtype=DataType.FLOAT32) -> Tensor:
    t = Tensor.randn(shape, dtype=dtype, device=device)
    return t


def _ref_softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------- Linear ----------------


@pytest.mark.parametrize(
    "m,k,out_f,bias", [(8, 16, 4, True), (64, 128, 32, False), (1, 1, 1, True)]
)
def test_linear_cpu(m, k, out_f, bias):
    x = np.random.randn(m, k).astype(np.float32)
    w = np.random.randn(out_f, k).astype(np.float32)
    b = np.random.randn(out_f).astype(np.float32)
    tx = Tensor.from_numpy(x)
    tw = Tensor.from_numpy(w)
    ref = x @ w.T + (b if bias else 0)
    if bias:
        out = axono.linear(tx, tw, Tensor.from_numpy(b))
    else:
        out = axono.linear_nobias(tx, tw)
    np.testing.assert_allclose(_to_numpy(out), ref, rtol=1e-4, atol=1e-4)


def test_linear_cpu_3d_broadcast_lastdim():
    """linear 支持任意前置维 (展开最后一维语义在归约类算子; linear 本身 2D)"""
    x = np.random.randn(4, 8).astype(np.float32)
    w = np.random.randn(3, 8).astype(np.float32)
    out = axono.linear(
        Tensor.from_numpy(x),
        Tensor.from_numpy(w),
        Tensor.zeros((3,), dtype=DataType.FLOAT32),
    )
    np.testing.assert_allclose(_to_numpy(out), x @ w.T, rtol=1e-4, atol=1e-4)


# ---------------- softmax / log_softmax ----------------


@pytest.mark.parametrize("shape", [(16,), (8, 16), (2, 3, 16)])
def test_softmax_cpu(shape):
    x = np.random.randn(*shape).astype(np.float32) * 3
    out = axono.softmax(Tensor.from_numpy(x))
    np.testing.assert_allclose(_to_numpy(out), _ref_softmax(x), rtol=1e-5, atol=1e-6)
    assert np.allclose(_to_numpy(out).sum(axis=-1), 1.0, atol=1e-6)


def test_log_softmax_cpu():
    x = np.random.randn(8, 16).astype(np.float32) * 3
    out = axono.log_softmax(Tensor.from_numpy(x))
    ref = (
        x
        - np.log(np.exp(x - x.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True))
        - x.max(axis=-1, keepdims=True)
    )
    np.testing.assert_allclose(_to_numpy(out), ref, rtol=1e-4, atol=1e-5)
    # 数值稳定: 大幅值不产生 nan
    big = np.array([[1000.0, 0.0]], dtype=np.float32)
    out_big = axono.log_softmax(Tensor.from_numpy(big))
    assert np.isfinite(_to_numpy(out_big)).all()


def test_softmax_overflow_stability_cpu():
    x = np.array([[1e4, 1e4 + 1], [0.0, 0.0]], dtype=np.float32)
    out = axono.softmax(Tensor.from_numpy(x))
    assert np.isfinite(_to_numpy(out)).all()
    np.testing.assert_allclose(
        _to_numpy(out)[0], [1 / (1 + math.e), math.e / (1 + math.e)], rtol=1e-5
    )


# ---------------- gelu / silu ----------------


def test_gelu_cpu():
    x = np.random.randn(64, 32).astype(np.float32)
    out = axono.gelu(Tensor.from_numpy(x))
    ref = 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2)))
    np.testing.assert_allclose(_to_numpy(out), ref, rtol=1e-5, atol=1e-6)


def test_silu_cpu():
    x = np.random.randn(64, 32).astype(np.float32)
    out = axono.silu(Tensor.from_numpy(x))
    ref = x / (1.0 + np.exp(-x))
    np.testing.assert_allclose(_to_numpy(out), ref, rtol=1e-5, atol=1e-6)


# ---------------- layer_norm / rms_norm ----------------


@pytest.mark.parametrize("shape", [(16,), (8, 16), (2, 4, 16)])
def test_layer_norm_cpu(shape):
    x = np.random.randn(*shape).astype(np.float32) * 2 + 1
    w = np.random.randn(shape[-1]).astype(np.float32)
    b = np.random.randn(shape[-1]).astype(np.float32)
    out = axono.layer_norm(
        Tensor.from_numpy(x), Tensor.from_numpy(w), Tensor.from_numpy(b)
    )
    flat = x.reshape(-1, shape[-1])
    mean = flat.mean(axis=-1, keepdims=True)
    var = flat.var(axis=-1, keepdims=True)
    ref = ((flat - mean) / np.sqrt(var + 1e-5) * w + b).reshape(shape)
    np.testing.assert_allclose(_to_numpy(out), ref, rtol=1e-4, atol=1e-5)


def test_rms_norm_cpu():
    x = np.random.randn(8, 16).astype(np.float32) * 2
    w = np.random.randn(16).astype(np.float32)
    out = axono.rms_norm(Tensor.from_numpy(x), Tensor.from_numpy(w))
    ref = x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-5) * w
    np.testing.assert_allclose(_to_numpy(out), ref, rtol=1e-4, atol=1e-5)


def test_layer_norm_normalization_cpu():
    """无仿射 (weight=1, bias=0) 时输出均值 0 方差 1"""
    x = np.random.randn(8, 16).astype(np.float32) * 3 + 5
    out = axono.layer_norm(
        Tensor.from_numpy(x),
        Tensor.ones((16,), dtype=DataType.FLOAT32),
        Tensor.zeros((16,), dtype=DataType.FLOAT32),
    )
    o = _to_numpy(out)
    np.testing.assert_allclose(o.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(o.var(axis=-1), 1.0, atol=1e-4)


# ---------------- CUDA ----------------


def test_linear_cuda(cuda_env):
    x = _randn((32, 64), "cuda")
    w = _randn((16, 64), "cuda")
    b = _randn((16,), "cuda")
    out = axono.linear(x, w, b)
    ref = x.to_numpy() @ w.to_numpy().T + b.to_numpy()
    np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-3, atol=1e-3)


def test_softmax_cuda(cuda_env):
    x = _randn((8, 64), "cuda")
    out = axono.softmax(x)
    np.testing.assert_allclose(
        out.to_numpy(), _ref_softmax(x.to_numpy()), rtol=1e-4, atol=1e-6
    )


def test_lmj_activations_cuda(cuda_env):
    x = _randn((32, 32), "cuda")
    g = axono.gelu(x).to_numpy()
    s = axono.silu(x).to_numpy()
    xn = x.to_numpy()
    np.testing.assert_allclose(
        g,
        0.5 * xn * (1 + np.vectorize(math.erf)(xn / math.sqrt(2))),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(s, xn / (1 + np.exp(-xn)), rtol=1e-4, atol=1e-5)


def test_norms_cuda(cuda_env):
    x = _randn((8, 32), "cuda")
    w = _randn((32,), "cuda")
    b = _randn((32,), "cuda")
    ln = axono.layer_norm(x, w, b).to_numpy()
    xn = x.to_numpy()
    mean = xn.mean(axis=-1, keepdims=True)
    var = xn.var(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        ln,
        (xn - mean) / np.sqrt(var + 1e-5) * w.to_numpy() + b.to_numpy(),
        rtol=1e-3,
        atol=1e-4,
    )
    rn = axono.rms_norm(x, w).to_numpy()
    np.testing.assert_allclose(
        rn,
        xn / np.sqrt((xn**2).mean(axis=-1, keepdims=True) + 1e-5) * w.to_numpy(),
        rtol=1e-3,
        atol=1e-4,
    )


def test_linear_graph_capture(cuda_env):
    """linear + softmax 可被 CUDA Graph 捕获且回放结果正确"""
    x = _randn((16, 32), "cuda")
    w = _randn((8, 32), "cuda")
    b = _randn((8,), "cuda")
    # 预跑一次确保句柄/workspace 就绪
    ref = axono.softmax(axono.linear(x, w, b)).to_numpy()
    with axono.cuda_graph() as g:
        g.output = axono.softmax(axono.linear(x, w, b))
    g.replay()
    g.sync()
    out = g.output.to_numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)
