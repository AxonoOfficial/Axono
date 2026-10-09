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
"""pytest 共享 fixture 与设备参数化。"""

import numpy as np
import pytest

import axono


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda: test requires a CUDA-capable device and CUDA build"
    )


def _devices():
    """可用设备列表 —— CPU 始终可用, CUDA 仅在编译+设备就绪时加入。"""
    devs = ["cpu"]
    if axono.cuda_available():
        try:
            axono.Tensor(axono.DataType.FLOAT32, [1], "cuda:0")
            devs.append("cuda:0")
        except Exception:  # pragma: no cover - 驱动/设备不可用
            pass
    return devs


DEVICES = _devices()


@pytest.fixture(params=DEVICES)
def device(request):
    """参数化 fixture: 在 cpu 与 (可用的) cuda:0 上各跑一遍。"""
    return request.param


@pytest.fixture
def cuda_env():
    if not axono.cuda_available():
        pytest.skip("需要 CUDA 构建")
    axono.set_backend("cuda")
    yield
    axono.set_backend("cpu")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def assert_allclose(tensor, array, rtol=1e-4, atol=1e-4):
    """把 axono.Tensor 与 numpy 数组对比 (自动处理设备拷回)。"""
    got = tensor.to_numpy()
    assert got.shape == array.shape, f"shape {got.shape} != {array.shape}"
    np.testing.assert_allclose(got, array, rtol=rtol, atol=atol)
