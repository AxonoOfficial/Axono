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
"""Python 后端控制: CPU/GPU 切换 + 全局设备设定。

用法::

    import axono
    axono.set_backend("cuda")        # 将张量工厂的默认设备设为 CUDA
    axono.get_backend()              # -> "cuda"
    axono.set_backend("cpu")         # 切回 CPU
    axono.set_backend("cuda:1")      # 指定 GPU 编号

    # matmul 底层实现控制 (仅 CUDA 后端有效):
    axono.set_matmul_backend("cublaslt")  # cuBLASLt (默认, 启发式选最优 kernel)
    axono.set_matmul_backend("cublas")    # 经典 cuBLAS

说明:
- set_backend 更新 set_default_device 管理的默认设备, 并额外检查后端名与 CUDA 构建支持。
- Tensor.randn/zeros/ones/full 未指定 device 时, 以及 Tensor.from_numpy
  创建的张量使用默认设备。算子输出跟随输入设备。
- 已有张量可用 tensor.to("cuda") 显式迁移; 本模块不做隐式迁移。
"""

from __future__ import annotations

import libaxono as _l

_VALID_BACKENDS = ("cpu", "cuda")


def set_backend(device: str) -> None:
    """设置全局后端/默认设备 ("cpu" | "cuda" | "cuda:<id>")。

    更新张量工厂的默认设备, 不迁移已有张量或模型。未启用 CUDA
    的构建上选择 cuda 会抛 RuntimeError; 此检查不确认驱动或 GPU 可用。

    示例::

        axono.set_backend("cuda")     # 后续张量工厂默认使用 CUDA
        axono.set_backend("cuda:1")   # 多卡指定卡号
        axono.set_backend("cpu")      # 切回 CPU
    """
    from .tensor import set_default_device

    if not isinstance(device, str) or not device:
        raise ValueError("backend 必须是非空字符串, 如 'cpu' / 'cuda' / 'cuda:0'")
    name = device.split(":")[0].lower()
    if name not in _VALID_BACKENDS:
        raise ValueError(f"未知后端 '{device}', 可选: cpu / cuda / cuda:<id>")
    if name == "cuda" and not _l.cuda_available():
        raise RuntimeError("当前构建未启用 CUDA, 无法切换到 GPU 后端")
    set_default_device(device)


def get_backend() -> str:
    """返回当前全局后端 ("cpu" / "cuda" / "cuda:<id>")。

    示例::

        axono.set_backend("cuda")
        assert axono.get_backend() == "cuda"
    """
    from .tensor import get_default_device

    return get_default_device()


def set_matmul_backend(name: str) -> None:
    """设置 CUDA matmul 底层实现 ("cublaslt" | "cublas")。

    FLOAT32/FLOAT64 可选 cuBLASLt 或经典 cuBLAS; cuBLASLt 失败时回退经典路径。
    FLOAT16 matmul 当前需要 cuBLASLt, INT32 使用内置 kernel。
    性能和浮点误差取决于设备、形状及所选算法。

    示例::

        axono.set_matmul_backend("cublaslt")   # 默认
        axono.set_matmul_backend("cublas")     # 切换后新 matmul 生效
    """
    if not _l.cuda_available():
        raise RuntimeError("当前构建未启用 CUDA")
    if name == "cublaslt":
        _l.use_cublas_lt(True)
    elif name == "cublas":
        _l.use_cublas_lt(False)
    else:
        raise ValueError(f"未知 matmul 后端 '{name}', 可选: cublaslt / cublas")


def get_matmul_backend() -> str:
    """返回当前 CUDA matmul 底层实现 ("cublaslt" | "cublas")。"""
    return "cublaslt" if _l.cublas_lt_enabled() else "cublas"
