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
    """设定全局默认设备。

    Tensor.randn/zeros/ones/full 未指定 device 时, 以及 Tensor.from_numpy
    创建的张量使用该设备。已有张量不受影响; 算子输出跟随输入设备。

    参数:
        device: "cpu" | "cuda" | "cuda:<id>" (如 "cuda:0" / "cuda:1")

    示例::

        import axono
        axono.set_default_device("cuda")   # 全局切到 GPU
        x = axono.Tensor.randn([1024, 1024])  # 使用默认 CUDA 设备

        axono.set_default_device("cuda:1") # 多卡时指定卡号
    """
    global _DEFAULT_DEVICE
    if not isinstance(device, str) or not device:
        raise ValueError("device 必须是非空字符串, 如 'cpu' / 'cuda' / 'cuda:0'")
    _DEFAULT_DEVICE = device


def get_default_device() -> str:
    """返回当前全局默认设备 (如 "cpu" / "cuda:0")。

    示例::

        axono.set_default_device("cuda")
        assert axono.get_default_device() == "cuda"
    """
    return _DEFAULT_DEVICE


# ---------------------------------------------------------------------------
# dtype 映射
# ---------------------------------------------------------------------------
_NUMPY_TO_DTYPE = {
    np.int8: DataType.INT8,
    np.int16: DataType.INT16,
    np.int32: DataType.INT32,
    np.int64: DataType.INT64,
    np.float16: DataType.FLOAT16,
    np.float32: DataType.FLOAT32,
    np.float64: DataType.FLOAT64,
    np.bool_: DataType.BOOLEAN,
}

_DTYPE_TO_ACCESSOR = {
    DataType.INT8: "data_int8",
    DataType.INT16: "data_int16",
    DataType.INT32: "data_int32",
    DataType.INT64: "data_int64",
    DataType.FLOAT16: "data_float16",
    DataType.FLOAT32: "data_float32",
    DataType.FLOAT64: "data_float64",
    DataType.BOOLEAN: "data_bool",
}


# ---------------------------------------------------------------------------
# 附加到绑定类上的便利方法
# ---------------------------------------------------------------------------
def _tensor_from_numpy(cls, array: np.ndarray) -> "Tensor":
    """从 numpy 数组创建 Tensor (拷贝数据)。

    非 C 连续数组自动 ascontiguousarray; 不支持的 dtype 自动转 float32。
    若已设置非 cpu 默认设备, 创建后自动迁移过去。当前不支持零维数组。
    float16 数组请先转 float32 再 cast_to(FLOAT16), 或使用 from_numpy_ref;
    直接 from_numpy(float16_array) 当前不能正确复制浮点位模式。

    参数:
        array: 至少一维的 numpy ndarray

    返回:
        Tensor: shape 与源数组一致, 位于默认设备上; 不支持的 dtype 转 FLOAT32

    示例::

        import numpy as np
        x = Tensor.from_numpy(np.random.rand(4, 8).astype(np.float32))
        y = Tensor.from_numpy(np.array([[1, 2], [3, 4]]))  # int → 保持 int
    """
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


def _tensor_from_numpy_ref(array: np.ndarray) -> "Tensor":
    """零拷贝借用 numpy 内存构造 CPU Tensor (不复制)。

    返回的 Tensor 直接引用 array 的内存 (不复制), 适合 mmap 权重加载:
    memmap → ref Tensor → to(cuda) 一步 DMA, 省去 host 中转拷贝。

    返回的 Tensor 保持源数组存活, 但其浅拷贝不会继承这层 Python 引用。
    派生浅拷贝仍在使用时, 请保留源数组。仅可向可写源数组写入;
    只读 memmap 借用的 Tensor 只能用于读取或复制。

    参数:
        array: C 连续的 float32 / float16 / int64 numpy ndarray 或 memmap
               (float16 内部以 uint16 视图借用)

    返回:
        Tensor: 位于 cpu, 与 array 共享存储

    示例::

        w = np.ones((4, 8), dtype=np.float32)
        t = Tensor.from_numpy_ref(w)      # 零拷贝
        g = t.to("cuda")                  # 一步 DMA 上卡
    """
    # np.asarray: memmap/子类 → 基类 ndarray 视图 (零拷贝), 让 nanobind
    # 的 nb::ndarray 签名能匹配。fp16 memmap (np.float16) 先零拷贝 view 成
    # uint16 (同字节), C++ 侧识别 uint16 → FLOAT16 借用。
    arr = np.asarray(array)
    if arr.dtype == np.float16:
        arr = arr.view(np.uint16)
    name = "from_borrowed_i64" if arr.dtype == np.int64 else "from_borrowed"
    return getattr(Tensor, name)(arr)


def _tensor_to_device(self, device: str) -> "Tensor":
    """设备迁移 (torch 语义)。

    同设备调用返回自身 (零拷贝, 不复制); 跨设备走 C++ 迁移, 返回新
    Tensor (原张量不变)。注意 C++ 按值返回经拷贝构造, 不会像旧版那样
    无谓双倍占显存。

    参数:
        device: 目标设备 "cpu" | "cuda" | "cuda:<id>"

    返回:
        Tensor: 同设备时为 self, 否则为位于目标设备的新 Tensor

    示例::

        x = Tensor.randn([1024, 1024])        # cpu
        g = x.to("cuda")                      # 上卡
        h = g.to("cuda")                      # 零拷贝, 即 g 自身
        back = g.to("cpu")                    # 拷回主机
    """
    if device == self.device:
        return self
    return _orig_to(self, device)


def _tensor_cast_to(self, target: "DataType") -> "Tensor":
    """dtype 类型转换 (返回新 Tensor)。

    CPU/CUDA 支持 FLOAT16 与 FLOAT32 互转, CPU 另支持 INT8 转 FLOAT32。
    同 dtype 时返回浅拷贝 (共享 storage); 其他转换尚未支持。
    FP16 权重占用为 FP32 的一半, CUDA Linear 可通过混合精度路径使用。

    参数:
        target: 目标 DataType (axono.DataType.FLOAT16 / FLOAT32 / ...)

    返回:
        Tensor: 转换后的新张量, 设备不变

    示例::

        from axono.core import DataType
        w16 = w32.cast_to(DataType.FLOAT16)   # fp32 → fp16
    """
    return self._orig_cast_to(DataType(target))


def _tensor_to_numpy(self) -> np.ndarray:
    """导出张量数据为 numpy 数组。

    CPU 返回共享内存视图, 修改一方另一方可见; 使用视图期间不要 resize
    或原地迁移原 Tensor。CUDA 自动拷回主机, 返回独立数组, 修改它不影响原 Tensor。
    FLOAT16 当前返回 int16 原始位模式, 需 .view(np.float16) 解释为浮点数。

    返回:
        np.ndarray: 除 FLOAT16 使用 int16 位模式外, dtype 与张量一致

    示例::

        x = Tensor.randn([2, 3], device="cpu")
        print(x.to_numpy().shape)      # (2, 3)
        x.to_numpy()[:] = 0            # 原地清零 (同时作用于张量)
        h = x.cast_to(DataType.FLOAT16)
        values = h.to_numpy().view(np.float16)
    """
    name = _DTYPE_TO_ACCESSOR.get(self.dtype)
    if name is None:
        raise ValueError(f"Unsupported dtype for numpy conversion: {self.dtype}")
    return getattr(self, name)()


def _tensor_copy_from_numpy(self, array: np.ndarray) -> "Tensor":
    """原地写入 numpy 数据 (形状与 dtype 须与当前张量完全匹配)。

    常用于把新 batch 写入 CUDA Graph 捕获的静态输入缓冲。

    参数:
        array: numpy ndarray, shape 与 dtype 必须与 self 一致

    抛出:
        ValueError: 形状或 dtype 不匹配

    示例::

        static_in = Tensor.zeros([8, 2048], device="cuda")
        static_in.copy_from_numpy(np.random.rand(8, 2048).astype(np.float32))
    """
    if list(array.shape) != list(self.shape):
        raise ValueError(
            f"copy_from_numpy: 形状不匹配 {list(array.shape)} vs {list(self.shape)}"
        )
    source = Tensor.from_numpy(np.ascontiguousarray(array))
    if source.dtype != self.dtype:
        raise ValueError(
            f"copy_from_numpy: dtype 不匹配 {source.dtype} vs {self.dtype}"
        )
    self.copy_from(source)
    return self


def _tensor_is_cuda(self) -> bool:
    return self.is_cuda


def _tensor_T(self) -> "Tensor":
    """2D 快捷转置 (等价 transpose(-2, -1))。

    示例::

        a = Tensor.randn([4, 8])
        b = a.T                       # shape (8, 4)
        c = a @ a.T                   # matmul 中常用
    """
    return self.transpose()


def _tensor_reshape(self, new_shape) -> "Tensor":
    """改形状 (原地, 支持 -1 自动推断)。

    元素总数不变; 至多一个维度可为 -1 (自动由其余维度推断)。

    参数:
        new_shape: 目标 shape (tuple/list), 可含一个 -1

    返回:
        Tensor: self (原地修改)

    抛出:
        ValueError: -1 多于一个, 或元素数无法整除
        RuntimeError: C++ 侧 reshape 失败

    示例::

        x = Tensor.randn([2, 3, 4])
        x.reshape([6, 4])     # (6, 4)
        x.reshape([-1, 8])    # (3, 8)  自动推断
    """
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


def _tensor_randn(
    shape,
    dtype: DataType = DataType.FLOAT32,
    device: str | None = None,
    mean: float = 0.0,
    stddev: float = 1.0,
) -> "Tensor":
    """创建正态分布随机张量 (可指定均值与标准差)。

    参数:
        shape: 目标 shape
        dtype: 数据类型, 默认 FLOAT32
        device: 目标设备, 缺省用全局默认设备
        mean / stddev: 正态分布均值与标准差, 默认 0 / 1

    示例::

        x = Tensor.randn([1024, 1024])                     # N(0,1) fp32
        y = Tensor.randn([8, 8], mean=0, stddev=0.02)      # 小方差初始化
    """
    return _orig["randn"](
        list(shape),
        dtype=dtype,
        device=device or get_default_device(),
        mean=mean,
        stddev=stddev,
    )


def _tensor_zeros(
    shape, dtype: DataType = DataType.FLOAT32, device: str | None = None
) -> "Tensor":
    """创建全 0 张量。

    示例::

        x = Tensor.zeros([8, 2048])                 # 默认 dtype/device
        y = Tensor.zeros([4, 4], device="cuda:0")   # 指定设备
    """
    tensor = Tensor(dtype, list(shape), device=device or get_default_device())
    tensor.fill_zero()
    return tensor


def _tensor_ones(
    shape, dtype: DataType = DataType.FLOAT32, device: str | None = None
) -> "Tensor":
    """创建全 1 张量。

    示例::

        x = Tensor.ones([3, 3], dtype=DataType.FLOAT32, device="cuda")
    """
    tensor = Tensor(dtype, list(shape), device=device or get_default_device())
    tensor.fill(1)
    return tensor


def _tensor_full(
    shape, value, dtype: DataType = DataType.FLOAT32, device: str | None = None
) -> "Tensor":
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
    Tensor.from_numpy_ref = staticmethod(_tensor_from_numpy_ref)
    Tensor.randn = staticmethod(_tensor_randn)
    Tensor.zeros = staticmethod(_tensor_zeros)
    Tensor.ones = staticmethod(_tensor_ones)
    Tensor.full = staticmethod(_tensor_full)
    Tensor.create_like = staticmethod(_tensor_create_like)
    # 实例方法
    Tensor.to_numpy = _tensor_to_numpy
    Tensor.cast_to = _tensor_cast_to
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
