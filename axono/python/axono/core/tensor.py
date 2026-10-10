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
Axono Tensor — v0.2 重构。

关键变化 (相对旧版):
- Tensor 直接就是 C++ 绑定类 (libaxono.Tensor), 不再有 Python 包装层,
  每次算子调用不再经过 from_raw() 二次封装 —— 绑定函数返回的对象
  即最终 Tensor, 零额外分配;
- 便利方法 (from_numpy / zeros / ones / T / 运算符等) 以猴子补丁方式
  附加到绑定类上, 实例方法直接访问底层 C++ 实现。

设备 API:
    axono.set_default_device("cpu" | "cuda" | "cuda:0")   # 全局默认设备
    Tensor(..., device=...)                               # 每张量显式指定
    tensor.to("cuda:1")                                   # 迁移
"""

from __future__ import annotations

import numpy as np
import libaxono as _l
from libaxono import DataType, Status

# ---------------------------------------------------------------------------
# 绑定类即公开类
# ---------------------------------------------------------------------------
Tensor = _l.Tensor

# ---------------------------------------------------------------------------
# 默认设备管理
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


# ---------------------------------------------------------------------------
# dtype 映射
# ---------------------------------------------------------------------------
_NUMPY_TO_DTYPE = {
    np.int8: DataType.INT8,
    np.int16: DataType.INT16,
    np.int32: DataType.INT32,
    np.int64: DataType.INT64,
    np.float32: DataType.FLOAT32,
    np.float64: DataType.FLOAT64,
    np.bool_: DataType.BOOLEAN,
}

_DTYPE_TO_ACCESSOR = {
    DataType.INT8: "data_int8",
    DataType.INT16: "data_int16",
    DataType.INT32: "data_int32",
    DataType.INT64: "data_int64",
    DataType.FLOAT32: "data_float32",
    DataType.FLOAT64: "data_float64",
    DataType.BOOLEAN: "data_bool",
}


# ---------------------------------------------------------------------------
# 附加到绑定类上的便利方法
# ---------------------------------------------------------------------------
def _tensor_from_numpy(cls, array: np.ndarray) -> "Tensor":
    """Create a Tensor from a numpy array (copies data)."""
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    dtype = _NUMPY_TO_DTYPE.get(array.dtype.type)
    if dtype is None:
        array = array.astype(np.float32)
        dtype = DataType.FLOAT32

    tensor = cls(dtype, list(array.shape), device="cpu")
    getattr(tensor, _DTYPE_TO_ACCESSOR[dtype])()[:] = array

    default = get_default_device()
    if default != "cpu":
        tensor = tensor.to(default)
    return tensor


def _tensor_to_device(self, device: str) -> "Tensor":
    """torch 语义: 同设备 to() 返回自身 (零拷贝); 跨设备走 C++ 迁移。"""
    if device == self.device:
        return self
    return _orig_to(self, device)


def _tensor_to_numpy(self) -> np.ndarray:
    """Return the tensor data as a numpy array."""
    name = _DTYPE_TO_ACCESSOR.get(self.dtype)
    if name is None:
        raise ValueError(f"Unsupported dtype for numpy conversion: {self.dtype}")
    return getattr(self, name)()


def _tensor_copy_from_numpy(self, array: np.ndarray) -> "Tensor":
    """原地写入 numpy 数据 (形状/类型须与当前张量匹配)。"""
    if list(array.shape) != list(self.shape):
        raise ValueError(
            f"copy_from_numpy: 形状不匹配 {list(array.shape)} vs {list(self.shape)}"
        )
    source = Tensor.from_numpy(np.ascontiguousarray(array))
    if source.dtype != self.dtype:
        raise ValueError(f"copy_from_numpy: dtype 不匹配 {source.dtype} vs {self.dtype}")
    self.copy_from(source)
    return self


def _tensor_is_cuda(self) -> bool:
    return self.is_cuda


def _tensor_T(self) -> "Tensor":
    """2D 快捷转置 (等价 transpose(-2, -1))"""
    return self.transpose()


def _tensor_reshape(self, new_shape) -> "Tensor":
    shape = [int(s) for s in new_shape]
    if any(s < 0 for s in shape):
        # 解析 -1: 由已知维与总元素数推
        known = 1
        n_neg = 0
        for s in shape:
            if s < 0:
                n_neg += 1
            else:
                known *= s
        if n_neg > 1:
            raise ValueError("reshape: 至多一个 -1")
        total = self.to_numpy().size if hasattr(self, "to_numpy") else None
        if total is None or total % known != 0:
            raise ValueError(f"reshape: 无法解析 -1 (known={known})")
        shape = [total // known if s < 0 else s for s in shape]
    st = _orig["reshape"](self, tuple(shape))
    if st != Status.OK:
        raise RuntimeError(f"Reshape failed with status: {st}")
    return self


def _tensor_resize(self, new_shape) -> "Tensor":
    st = _orig["resize"](self, list(new_shape))
    if st != Status.OK:
        raise RuntimeError(f"Resize failed with status: {st}")
    return self


def _tensor_fill_zero(self) -> "Tensor":
    st = _orig["fill_zero"](self)
    if st != Status.OK:
        raise RuntimeError(f"Fill zero failed with status: {st}")
    return self


def _tensor_fill(self, value) -> "Tensor":
    st = _orig["fill"](self, value)
    if st != Status.OK:
        raise RuntimeError(f"Fill failed with status: {st}")
    return self


def _tensor_matmul(self, other) -> "Tensor":
    from .operators import matmul

    return matmul(self, other)


def _tensor_add(self, other) -> "Tensor":
    from .operators import add

    return add(self, other)


def _tensor_sub(self, other) -> "Tensor":
    from .operators import sub

    return sub(self, other)


def _tensor_mul(self, other) -> "Tensor":
    from .operators import mul

    return mul(self, other)


def _tensor_truediv(self, other) -> "Tensor":
    from .operators import div

    return div(self, other)


def _tensor_neg(self) -> "Tensor":
    from .ops import neg

    return neg(self)


def _tensor_abs(self) -> "Tensor":
    from .ops import abs as _abs

    return _abs(self)


def _tensor_is_same_shape(self, other: "Tensor") -> bool:
    return self.is_same_shape(other)


def _tensor_randn(shape, dtype: DataType = DataType.FLOAT32,
                  device: str | None = None, mean: float = 0.0,
                  stddev: float = 1.0) -> "Tensor":
    return _orig["randn"](
        list(shape),
        dtype=dtype,
        device=device or get_default_device(),
        mean=mean,
        stddev=stddev,
    )


def _tensor_zeros(shape, dtype: DataType = DataType.FLOAT32,
                  device: str | None = None) -> "Tensor":
    tensor = Tensor(dtype, list(shape), device=device or get_default_device())
    tensor.fill_zero()
    return tensor


def _tensor_ones(shape, dtype: DataType = DataType.FLOAT32,
                 device: str | None = None) -> "Tensor":
    tensor = Tensor(dtype, list(shape), device=device or get_default_device())
    tensor.fill(1)
    return tensor


def _tensor_full(shape, value, dtype: DataType = DataType.FLOAT32,
                 device: str | None = None) -> "Tensor":
    tensor = Tensor(dtype, list(shape), device=device or get_default_device())
    tensor.fill(value)
    return tensor


def _tensor_create_like(other: "Tensor") -> "Tensor":
    return _l.Tensor.create_like(other)


def _attach() -> None:
    """把便利方法挂到绑定类上 (幂等)。"""
    orig = {
        "to": _l.Tensor.to,
        "reshape": _l.Tensor.reshape,
        "resize": _l.Tensor.resize,
        "fill_zero": _l.Tensor.fill_zero,
        "fill": _l.Tensor.fill,
        "randn": _l.Tensor.randn,
    }
    globals()["_orig"] = orig
    globals()["_orig_to"] = orig["to"]

    # classmethod / staticmethod 工厂
    Tensor.from_numpy = classmethod(_tensor_from_numpy)
    Tensor.randn = staticmethod(_tensor_randn)
    Tensor.zeros = staticmethod(_tensor_zeros)
    Tensor.ones = staticmethod(_tensor_ones)
    Tensor.full = staticmethod(_tensor_full)
    Tensor.create_like = staticmethod(_tensor_create_like)
    # 实例方法
    Tensor.to_numpy = _tensor_to_numpy
    Tensor.copy_from_numpy = _tensor_copy_from_numpy
    # torch 语义: 同设备 to() 返回自身 (零拷贝)。C++ 按值返回会经拷贝构造
    # 深拷贝一份 (Linear 构建期 from_numpy→to(device) 曾因此双倍显存)。
    Tensor.to = _tensor_to_device
    Tensor.T = property(_tensor_T)
    Tensor.reshape = _tensor_reshape
    Tensor.resize = _tensor_resize
    Tensor.fill_zero = _tensor_fill_zero
    Tensor.fill = _tensor_fill
    # 运算符
    Tensor.__matmul__ = _tensor_matmul
    Tensor.__add__ = _tensor_add
    Tensor.__radd__ = _tensor_add
    Tensor.__sub__ = _tensor_sub
    Tensor.__mul__ = _tensor_mul
    Tensor.__truediv__ = _tensor_truediv
    Tensor.__neg__ = _tensor_neg
    Tensor.__abs__ = _tensor_abs


_attach()
