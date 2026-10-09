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
"""Tensor 基础行为测试 (创建、属性、转置、填充、numpy 互操作)。"""

import numpy as np
import pytest

from axono.core import DataType, Tensor

from ..conftest import assert_allclose


class TestCreation:
    def test_creation_shapes(self):
        for shape in [[1], [2, 3], [1, 1, 1], [2, 2, 2, 2]]:
            t = Tensor(shape=shape)
            assert t.shape == shape

    def test_is_same_shape(self):
        a = Tensor(DataType.FLOAT32, [2, 3])
        b = Tensor(DataType.FLOAT32, [2, 3])
        c = Tensor(DataType.FLOAT32, [3, 2])
        assert a.is_same_shape(b)
        assert not a.is_same_shape(c)

    def test_properties(self, device):
        t = Tensor(DataType.FLOAT32, [2, 3], device=device)
        assert t.shape == [2, 3]
        assert t.ndim == 2
        assert t.num_elements == 6
        assert t.num_bytes == 24
        assert isinstance(t.device, str)
        assert t.dtype == DataType.FLOAT32

    @pytest.mark.parametrize(
        "dtype,dt_size",
        [
            (DataType.FLOAT32, 4),
            (DataType.FLOAT64, 8),
            (DataType.INT32, 4),
            (DataType.INT64, 8),
            (DataType.INT16, 2),
            (DataType.INT8, 1),
            (DataType.BOOLEAN, 1),
        ],
    )
    def test_dtypes(self, dtype, dt_size, device):
        t = Tensor(dtype, [2, 2], device=device)
        assert t.dtype == dtype
        assert t.num_bytes == 4 * dt_size


class TestTransforms:
    def test_transpose(self, device):
        t = Tensor(DataType.FLOAT32, [2, 3], device=device)
        t.fill(1.0)
        assert t.transpose().shape == [3, 2]

    def test_transpose_negative_dims(self, device):
        t = Tensor(DataType.FLOAT32, [2, 3, 4], device=device)
        assert t.transpose(0, -1).shape == [4, 3, 2]
        assert t.transpose(-3, -2).shape == [3, 2, 4]

    def test_transpose_invalid(self, device):
        t = Tensor(DataType.FLOAT32, [2, 3], device=device)
        with pytest.raises(Exception):
            t.transpose(0, 5)

    def test_reshape(self, device):
        t = Tensor(DataType.FLOAT32, [2, 3], device=device)
        t.fill(1.0)
        r = t.reshape([3, 2])
        assert r.shape == [3, 2]
        assert r.to_numpy().sum() == 6.0

    def test_reshape_bad_size(self, device):
        t = Tensor(DataType.FLOAT32, [2, 3], device=device)
        with pytest.raises(Exception):
            t.reshape([4, 4])


class TestFill:
    def test_fill_float(self, device):
        t = Tensor(DataType.FLOAT32, [1, 3], device=device)
        t.fill(3.14)
        assert_allclose(t, np.full((1, 3), 3.14, dtype=np.float32))

    def test_fill_int(self, device):
        t = Tensor(DataType.INT32, [2, 2], device=device)
        t.fill(42)
        np.testing.assert_array_equal(t.to_numpy(), np.full((2, 2), 42, np.int32))

    def test_fill_zero(self, device):
        t = Tensor(DataType.FLOAT32, [3, 3], device=device)
        t.fill_zero()
        np.testing.assert_array_equal(t.to_numpy(), np.zeros((3, 3), np.float32))

    def test_factories(self, device):
        z = Tensor.zeros([2, 2], device=device)
        o = Tensor.ones([2, 2], device=device)
        f = Tensor.full([2, 2], 7, device=device)
        np.testing.assert_array_equal(z.to_numpy(), np.zeros((2, 2), np.float32))
        np.testing.assert_array_equal(o.to_numpy(), np.ones((2, 2), np.float32))
        np.testing.assert_array_equal(f.to_numpy(), np.full((2, 2), 7, np.float32))


class TestNumpyInterop:
    def test_roundtrip_float32(self, device):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = Tensor.from_numpy(arr).to(device)
        np.testing.assert_array_equal(t.to_numpy(), arr)

    def test_roundtrip_dtypes(self, device):
        for np_dtype, in [
            (np.int8,), (np.int16,), (np.int32,), (np.int64,),
            (np.float32,), (np.float64,),
        ]:
            arr = (np.arange(6) + 1).astype(np_dtype).reshape(2, 3)
            t = Tensor.from_numpy(arr).to(device)
            np.testing.assert_array_equal(t.to_numpy(), arr)

    def test_bool(self, device):
        arr = np.array([[True, False], [False, True]])
        t = Tensor.from_numpy(arr).to(device)
        np.testing.assert_array_equal(t.to_numpy(), arr)

    def test_non_contiguous_input(self, device):
        base = np.arange(20, dtype=np.float32).reshape(4, 5)
        view = base[:, ::2]  # 非连续视图
        t = Tensor.from_numpy(view).to(device)
        np.testing.assert_array_equal(t.to_numpy(), np.ascontiguousarray(view))

    def test_randn_stats(self, device):
        t = Tensor.randn([2000], device=device)
        data = t.to_numpy()
        assert data.shape == (2000,)
        assert abs(data.mean()) < 0.2
        assert 0.5 < data.std() < 1.6


class TestDevices:
    def test_to_cpu_roundtrip(self, device):
        t = Tensor(DataType.FLOAT32, [2, 2], device=device)
        t.fill(9.0)
        back = t.to("cpu")
        assert back.device == "cpu"
        np.testing.assert_array_equal(back.to_numpy(), np.full((2, 2), 9, np.float32))

    @pytest.mark.cuda
    def test_to_cuda(self):
        if not all("cuda" in d for d in ["cpu"]):  # placeholder guard
            pass
