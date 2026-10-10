"""FP16 混合推理: CastTo + fp16 Linear tensor-core 路径测试。"""

import numpy as np
import pytest

import axono
from axono import Tensor
from axono.core.tensor import DataType


def test_cast_f32_to_f16_cpu():
    x = np.array([1.0, -0.5, 65504.0, 0.0], dtype=np.float32)
    t = Tensor.from_numpy(x)
    t16 = t.cast_to(DataType.FLOAT16)
    assert t16.dtype == DataType.FLOAT16
    back = t16.cast_to(DataType.FLOAT32).to_numpy()
    np.testing.assert_allclose(back, x, rtol=1e-3)


def test_cast_f16_roundtrip_random():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    t = Tensor.from_numpy(x)
    back = t.cast_to(DataType.FLOAT16).cast_to(DataType.FLOAT32).to_numpy()
    np.testing.assert_allclose(back, x, rtol=1e-2, atol=1e-3)


def test_linear_fp16_matches_fp32():
    """fp16 混合 Linear 与 fp32 参考一致性 (rtol ~1e-2)。"""
    if not axono.cuda_available():
        pytest.skip("需要 CUDA")
    from axono import nn

    axono.set_backend("cuda")
    rng = np.random.default_rng(1)
    x_np = rng.normal(size=(8, 64)).astype(np.float32)
    w_np = rng.normal(size=(32, 64)).astype(np.float32) * 0.05
    b_np = rng.normal(size=(32,)).astype(np.float32) * 0.05

    lin = nn.Linear(64, 32, device="cuda")
    lin.load_state_dict({"weight": w_np, "bias": b_np})
    ref = lin(Tensor.from_numpy(x_np).to("cuda")).to_numpy()

    lin.cast_linear_fp16()
    out16 = lin(Tensor.from_numpy(x_np).to("cuda")).to_numpy()
    np.testing.assert_allclose(out16, ref, rtol=5e-2, atol=5e-2)


def test_borrowed_f16_zero_copy():
    """from_numpy_ref 支持 fp16 (uint16 位模式借用)。"""
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    t = Tensor.from_numpy_ref(x)
    assert t.dtype == DataType.FLOAT16
    back = t.cast_to(DataType.FLOAT32).to_numpy()
    np.testing.assert_allclose(back, x.astype(np.float32), rtol=1e-3)
