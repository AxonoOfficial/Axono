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
"""
Axono Tensor — Python interface for the C++ Tensor (v0.2, nanobind).

设备 API:
    axono.set_default_device("cpu" | "cuda" | "cuda:0")   # 全局默认设备
    Tensor(..., device=...)                               # 每张量显式指定
    tensor.to("cuda:1")                                   # 迁移
"""

from __future__ import annotations

import numpy as np
from libaxono import DataType, Status
from libaxono import Tensor as _Tensor

# ---------------------------------------------------------------------------
# 默认设备管理 (取代散落各处的 os.getenv("axono_default_device") 读取)
# ---------------------------------------------------------------------------
_DEFAULT_DEVICE = "cpu"


def set_default_device(device: str) -> None:
    """Set the global default device ("cpu", "cuda" or "cuda:<id>")."""
    global _DEFAULT_DEVICE
    if not isinstance(device, str) or not device:
        raise ValueError("device 必须是非空字符串, 如 'cpu' / 'cuda' / 'cuda:0'")
    _DEFAULT_DEVICE = device


def get_default_device() -> str:
    """Return the current default device."""
    return _DEFAULT_DEVICE


class Tensor:
    """Python Tensor class wrapping the C++ Tensor."""

    def __init__(
        self,
        dtype: DataType = DataType.FLOAT32,
        shape: list | tuple | None = None,
        device: str | None = None,
    ):
        if shape is None:
            self._tensor = _Tensor(dtype)
        else:
            self._tensor = _Tensor(
                dtype, list(shape), device or get_default_device()
            )

    # ------------------------------------------------------------------
    # 构造工具
    # ------------------------------------------------------------------
    @classmethod
    def create_like(cls, other: "Tensor") -> "Tensor":
        """Create a tensor with the same shape and dtype as ``other``."""
        return cls.from_raw(_Tensor.create_like(other._tensor))

    @classmethod
    def from_raw(cls, raw_tensor) -> "Tensor":
        obj = cls.__new__(cls)
        obj._tensor = raw_tensor
        return obj

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "Tensor":
        """Create a Tensor from a numpy array (copies data)."""
        dtype_map = {
            np.int8: DataType.INT8,
            np.int16: DataType.INT16,
            np.int32: DataType.INT32,
            np.int64: DataType.INT64,
            np.float32: DataType.FLOAT32,
            np.float64: DataType.FLOAT64,
            np.bool_: DataType.BOOLEAN,
        }
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        dtype = dtype_map.get(array.dtype.type)
        if dtype is None:
            # 不认识的类型统一转 float32
            array = array.astype(np.float32)
            dtype = DataType.FLOAT32

        tensor = cls(dtype, list(array.shape), device="cpu")
        accessor = tensor._tensor

        if dtype == DataType.INT8:
            accessor.data_int8()[:] = array
        elif dtype == DataType.INT16:
            accessor.data_int16()[:] = array
        elif dtype == DataType.INT32:
            accessor.data_int32()[:] = array
        elif dtype == DataType.INT64:
            accessor.data_int64()[:] = array
        elif dtype == DataType.FLOAT32:
            accessor.data_float32()[:] = array
        elif dtype == DataType.FLOAT64:
            accessor.data_float64()[:] = array
        elif dtype == DataType.BOOLEAN:
            accessor.data_bool()[:] = array

        # 与全局默认设备保持一致 (from_numpy 恒先落 CPU, 再按需迁移)
        default = get_default_device()
        if default != "cpu":
            tensor = tensor.to(default)
        return tensor

    # ------------------------------------------------------------------
    # 设备 / 形状
    # ------------------------------------------------------------------
    def is_cuda(self) -> bool:
        return self._tensor.is_cuda

    def to(self, device: str) -> "Tensor":
        return Tensor.from_raw(self._tensor.to(device))

    def transpose(self, dim0: int = -2, dim1: int = -1) -> "Tensor":
        return Tensor.from_raw(self._tensor.transpose(dim0, dim1))

    @property
    def T(self) -> "Tensor":
        """2D 快捷转置 (等价 transpose(-2, -1))"""
        return self.transpose()

    def reshape(self, new_shape) -> "Tensor":
        status = self._tensor.reshape(list(new_shape))
        if status != Status.OK:
            raise RuntimeError(f"Reshape failed with status: {status}")
        return self

    def resize(self, new_shape) -> "Tensor":
        status = self._tensor.resize(list(new_shape))
        if status != Status.OK:
            raise RuntimeError(f"Resize failed with status: {status}")
        return self

    # ------------------------------------------------------------------
    # 填充
    # ------------------------------------------------------------------
    def fill_zero(self) -> "Tensor":
        status = self._tensor.fill_zero()
        if status != Status.OK:
            raise RuntimeError(f"Fill zero failed with status: {status}")
        return self

    def fill(self, value) -> "Tensor":
        status = self._tensor.fill(value)
        if status != Status.OK:
            raise RuntimeError(f"Fill failed with status: {status}")
        return self

    # ------------------------------------------------------------------
    # 运算符
    # ------------------------------------------------------------------
    def __matmul__(self, other) -> "Tensor":
        from .operators import matmul

        return matmul(self, other)

    def __add__(self, other) -> "Tensor":
        from .operators import add

        return add(self, other)

    __radd__ = __add__
    __rmatmul__ = __matmul__

    # ------------------------------------------------------------------
    # 数据交互
    # ------------------------------------------------------------------
    def to_numpy(self) -> np.ndarray:
        """Return the tensor data as a numpy array."""
        accessors = {
            DataType.INT8: "data_int8",
            DataType.INT16: "data_int16",
            DataType.INT32: "data_int32",
            DataType.INT64: "data_int64",
            DataType.FLOAT32: "data_float32",
            DataType.FLOAT64: "data_float64",
            DataType.BOOLEAN: "data_bool",
        }
        name = accessors.get(self.dtype)
        if name is None:
            raise ValueError(f"Unsupported dtype for numpy conversion: {self.dtype}")
        return getattr(self._tensor, name)()

    def copy_from_numpy(self, array: np.ndarray) -> "Tensor":
        """原地写入 numpy 数据 (形状/类型须与当前张量匹配)。"""
        if list(array.shape) != self.shape:
            raise ValueError(
                f"copy_from_numpy: 形状不匹配 {list(array.shape)} vs {self.shape}"
            )
        source = Tensor.from_numpy(np.ascontiguousarray(array))
        if source.dtype != self.dtype:
            raise ValueError(f"copy_from_numpy: dtype 不匹配 {source.dtype} vs {self.dtype}")
        self._tensor.copy_from(source._tensor)
        return self

    def is_same_shape(self, other: "Tensor") -> bool:
        return self._tensor.is_same_shape(other._tensor)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def dtype(self) -> DataType:
        return self._tensor.dtype

    @property
    def device(self) -> str:
        return self._tensor.device

    @property
    def shape(self) -> list:
        return list(self._tensor.shape)

    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    @property
    def num_elements(self) -> int:
        return self._tensor.num_elements

    @property
    def num_bytes(self) -> int:
        return self._tensor.num_bytes

    def __repr__(self) -> str:
        return self._tensor.__repr__()

    def __str__(self) -> str:
        return self._tensor.__str__()

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @staticmethod
    def randn(
        shape,
        dtype: DataType = DataType.FLOAT32,
        device: str | None = None,
        mean: float = 0.0,
        stddev: float = 1.0,
    ) -> "Tensor":
        return Tensor.from_raw(
            _Tensor.randn(
                list(shape),
                dtype=dtype,
                device=device or get_default_device(),
                mean=mean,
                stddev=stddev,
            )
        )

    @staticmethod
    def zeros(
        shape, dtype: DataType = DataType.FLOAT32, device: str | None = None
    ) -> "Tensor":
        tensor = Tensor(dtype, shape, device=device)
        tensor.fill_zero()
        return tensor

    @staticmethod
    def ones(
        shape, dtype: DataType = DataType.FLOAT32, device: str | None = None
    ) -> "Tensor":
        tensor = Tensor(dtype, shape, device=device)
        tensor.fill(1)
        return tensor

    @staticmethod
    def full(
        shape,
        value,
        dtype: DataType = DataType.FLOAT32,
        device: str | None = None,
    ) -> "Tensor":
        tensor = Tensor(dtype, shape, device=device)
        tensor.fill(value)
        return tensor
