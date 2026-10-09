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
"""后端控制 (set_backend / set_matmul_backend) 测试。"""

import numpy as np
import pytest

import axono


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    axono.set_backend("cpu")


def test_set_backend_cpu():
    axono.set_backend("cpu")
    assert axono.get_backend() == "cpu"
    t = axono.Tensor.randn([4, 4])
    assert not t.is_cuda


def test_set_backend_invalid():
    with pytest.raises(ValueError):
        axono.set_backend("tpu")
    with pytest.raises(ValueError):
        axono.set_backend("")


def test_backend_new_tensors_follow():
    axono.set_backend("cpu")
    a = axono.Tensor.zeros([2, 3])
    assert not a.is_cuda
    axono.set_backend("cpu")
    b = axono.Tensor.randn([2, 3])
    assert not b.is_cuda


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_set_backend_cuda_and_ops():
    axono.set_backend("cuda")
    assert axono.get_backend().startswith("cuda")
    a = axono.Tensor.randn([8, 8])
    b = axono.Tensor.randn([8, 8])
    assert a.is_cuda and b.is_cuda
    c = axono.matmul(a, b)
    assert c.is_cuda
    # 经 cpu 取回校验
    cpu_a, cpu_b, cpu_c = a.to("cpu"), b.to("cpu"), c.to("cpu")
    np.testing.assert_allclose(
        cpu_c.to_numpy(), cpu_a.to_numpy() @ cpu_b.to_numpy(), rtol=1e-4, atol=1e-4
    )
    axono.set_backend("cpu")


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_matmul_backend_switch():
    axono.set_backend("cuda")
    assert axono.get_matmul_backend() == "cublaslt"
    a = axono.Tensor.randn([16, 16])
    b = axono.Tensor.randn([16, 16])
    axono.set_matmul_backend("cublaslt")
    c_lt = axono.matmul(a, b).to("cpu").to_numpy()
    axono.set_matmul_backend("cublas")
    assert axono.get_matmul_backend() == "cublas"
    c_blas = axono.matmul(a, b).to("cpu").to_numpy()
    np.testing.assert_allclose(c_lt, c_blas, rtol=1e-4, atol=1e-4)
    axono.set_matmul_backend("cublaslt")
    with pytest.raises(ValueError):
        axono.set_matmul_backend("mkl")
    axono.set_backend("cpu")


@pytest.mark.skipif(not axono.cuda_available(), reason="需要 CUDA 构建")
def test_matmul_backend_all_dtypes():
    """int 路径不受 Lt 开关影响。"""
    axono.set_backend("cuda")
    a = axono.Tensor.from_numpy(np.arange(12, dtype=np.int32).reshape(3, 4))
    b = axono.Tensor.from_numpy(np.ones((4, 2), dtype=np.int32))
    for backend in ("cublaslt", "cublas"):
        axono.set_matmul_backend(backend)
        c = axono.matmul(a, b)
        assert list(c.to_numpy().flatten()) == [6, 6, 22, 22, 38, 38]
    axono.set_backend("cpu")
