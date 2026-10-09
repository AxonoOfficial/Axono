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
"""二元逐元素算子: sub/mul/div (同形, CPU/CUDA 后端自动分派)。

与 add 同风格: 非原地保证输入不变; out= 走 into 路径 (零额外分配)。
"""

import libaxono as _l

from ..tensor import Tensor  # noqa: F401 — 绑定类, Pyright 无法解析


def _check(a, b, name):  # noqa: ANN001
    if a.device != b.device:
        raise ValueError(f"{name}: 输入张量不在同一设备上 ({a.device} vs {b.device})")
    if a.dtype != b.dtype:
        raise ValueError(f"{name}: 数据类型不一致 ({a.dtype} vs {b.dtype})")
    if a.shape != b.shape:
        raise ValueError(f"{name}: 形状不匹配 {a.shape} vs {b.shape}")


def _binary(a, b, name, fn, out):  # noqa: ANN001
    _check(a, b, name)
    if out is not None:
        _check(out, a, name + ": out")
        result = fn(a, b)          # 计算到新张量
        out.copy_from(result)      # 拷入 out (设备间安全)
        return out
    return fn(a, b)


def sub(a, b, out=None):  # noqa: ANN001
    """逐元素减法 a - b (同形)。"""
    return _binary(a, b, "sub", _l.sub, out)


def mul(a, b, out=None):  # noqa: ANN001
    """逐元素乘法 a * b (同形)。"""
    return _binary(a, b, "mul", _l.mul, out)


def div(a, b, out=None):  # noqa: ANN001
    """逐元素除法 a / b (同形, b=0 时为 inf/nan, 与 torch 一致)。"""
    return _binary(a, b, "div", _l.div, out)


__all__ = ["sub", "mul", "div"]
