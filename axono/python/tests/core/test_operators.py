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
"""算子测试: add / matmul / relu (CPU + CUDA/cuBLAS)。"""

import numpy as np
import pytest

import axono
from axono.core import DataType, Tensor
from axono.core.operators import add, matmul
from axono.core.ops import relu


class TestAdd:
    def test_add_basic(self, device):
        a = Tensor.from_numpy(np.full((2, 3), 1.5, np.float32)).to(device)
        b = Tensor.from_numpy(np.full((2, 3), 2.5, np.float32)).to(device)
        np.testing.assert_allclose(
            add(a, b).to_numpy(), np.full((2, 3), 4.0, np.float32)
        )

    def test_add_operator(self, device):
        a = Tensor.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3)).to(device)
        b = Tensor.from_numpy(np.ones((2, 3), np.float32)).to(device)
        np.testing.assert_allclose(
            (a + b).to_numpy(), np.arange(6, dtype=np.float32).reshape(2, 3) + 1
        )

    def test_add_shape_mismatch(self, device):
        a = Tensor(DataType.FLOAT32, [2, 3], device=device)
        b = Tensor(DataType.FLOAT32, [3, 2], device=device)
        with pytest.raises(Exception):
            add(a, b)


class TestMatMul:
    @pytest.mark.parametrize(
        "m,k,n",
        [(4, 4, 4), (3, 5, 7), (1, 1, 1), (16, 32, 8), (64, 64, 64), (128, 256, 64)],
    )
    def test_matmul_shapes(self, m, k, n, device, rng):
        na = rng.standard_normal((m, k)).astype(np.float32)
        nb = rng.standard_normal((k, n)).astype(np.float32)
        a = Tensor.from_numpy(na).to(device)
        b = Tensor.from_numpy(nb).to(device)
        got = (a @ b).to_numpy()
        np.testing.assert_allclose(got, na @ nb, rtol=1e-3, atol=1e-3)

    def test_matmul_operator_equivalence(self, device, rng):
        na = rng.standard_normal((8, 4)).astype(np.float32)
        nb = rng.standard_normal((4, 6)).astype(np.float32)
        a = Tensor.from_numpy(na).to(device)
        b = Tensor.from_numpy(nb).to(device)
        np.testing.assert_allclose(
            matmul(a, b).to_numpy(), (a @ b).to_numpy(), rtol=1e-5
        )

    def test_matmul_float64(self, device, rng):
        na = rng.standard_normal((5, 3)).astype(np.float64)
        nb = rng.standard_normal((3, 4)).astype(np.float64)
        a = Tensor.from_numpy(na).to(device)
        b = Tensor.from_numpy(nb).to(device)
        np.testing.assert_allclose((a @ b).to_numpy(), na @ nb, rtol=1e-6)

    def test_matmul_shape_mismatch(self, device):
        a = Tensor(DataType.FLOAT32, [2, 3], device=device)
        b = Tensor(DataType.FLOAT32, [2, 3], device=device)
        with pytest.raises(Exception):
            matmul(a, b)

    def test_matmul_large_cublas(self, device, rng):
        """较大规模, 主要用于验证 CUDA 下 cuBLAS path。"""
        if device == "cpu":
            pytest.skip("CPU 下由 AVX kernel 覆盖, 大矩阵交给 CUDA 用例")
        na = rng.standard_normal((256, 512)).astype(np.float32)
        nb = rng.standard_normal((512, 128)).astype(np.float32)
        a = Tensor.from_numpy(na).to(device)
        b = Tensor.from_numpy(nb).to(device)
        np.testing.assert_allclose((a @ b).to_numpy(), na @ nb, rtol=1e-2, atol=1e-2)


class TestRelu:
    def test_relu_basic(self, device, rng):
        arr = rng.standard_normal((4, 5)).astype(np.float32)
        t = Tensor.from_numpy(arr).to(device)
        np.testing.assert_allclose(relu(t).to_numpy(), np.maximum(arr, 0))

    def test_relu_all_negative(self, device):
        arr = -np.ones((3, 3), np.float32)
        t = Tensor.from_numpy(arr).to(device)
        np.testing.assert_array_equal(relu(t).to_numpy(), np.zeros((3, 3), np.float32))

    def test_relu_inplace(self, device, rng):
        arr = rng.standard_normal((4, 4)).astype(np.float32)
        t = Tensor.from_numpy(arr).to(device)
        out = relu(t, inplace=True)
        np.testing.assert_allclose(out.to_numpy(), np.maximum(arr, 0))

    def test_relu_zero(self, device):
        t = Tensor.zeros([5], device=device)
        np.testing.assert_array_equal(relu(t).to_numpy(), np.zeros(5, np.float32))


class TestRandnOp:
    def test_randn_shape_and_dtype(self, device):
        t = Tensor.randn([3, 4], device=device)
        assert t.shape == [3, 4]
        assert t.dtype == DataType.FLOAT32

    def test_randn_reproducible_stats(self, device):
        t = Tensor.randn([5000], mean=2.0, stddev=0.5, device=device)
        d = t.to_numpy()
        assert abs(d.mean() - 2.0) < 0.1
        assert abs(d.std() - 0.5) < 0.1


class TestDeviceConsistency:
    """同一运算在 CPU 与 CUDA 上结果应一致 (设备参数化时自动比对)。"""

    def test_matmul_device_parity(self, device, rng):
        na = rng.standard_normal((32, 48)).astype(np.float32)
        nb = rng.standard_normal((48, 24)).astype(np.float32)
        a = Tensor.from_numpy(na).to(device)
        b = Tensor.from_numpy(nb).to(device)
        expected = na @ nb
        np.testing.assert_allclose((a @ b).to_numpy(), expected, rtol=1e-3, atol=1e-3)
