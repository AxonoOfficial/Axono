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
"""逐元素一元算子: neg/abs/exp/log/sqrt/sigmoid/tanh。

全部走 libaxono 绑定 (CPU/CUDA 后端自动分派), 非原地, 保证输入不变。
"""

import libaxono as _l

from ..tensor import Tensor  # noqa: F401 — 供类型提示 (绑定类, Pyright 无法解析)


def neg(x):  # noqa: ANN001 — x: Tensor
    """-x"""
    return _l.neg(x)


def abs(x):  # noqa: A001 — 与 torch.abs 同名
    """逐元素绝对值"""
    return _l.abs(x)


def exp(x):
    """逐元素自然指数 e^x"""
    return _l.exp(x)


def log(x):
    """逐元素自然对数 ln(x) (x<=0 时为 -inf/nan, 与 torch 一致)"""
    return _l.log(x)


def sqrt(x):
    """逐元素平方根"""
    return _l.sqrt(x)


def sigmoid(x):
    """Sigmoid 激活 1/(1+e^-x)"""
    return _l.sigmoid(x)


def tanh(x):
    """双曲正切"""
    return _l.tanh(x)


def sin(x):
    """逐元素正弦"""
    return _l.sin(x)


def cos(x):
    """逐元素余弦"""
    return _l.cos(x)


def rsqrt(x):
    """逐元素平方根倒数 1/sqrt(x)"""
    return _l.rsqrt(x)


def square(x):
    """逐元素平方 x*x"""
    return _l.square(x)


def reciprocal(x):
    """逐元素倒数 1/x"""
    return _l.reciprocal(x)


def sign(x):
    """逐元素符号 (-1/0/1)"""
    return _l.sign(x)


def floor(x):
    """逐元素向下取整"""
    return _l.floor(x)


def ceil(x):
    """逐元素向上取整"""
    return _l.ceil(x)


def round(x):  # noqa: A001 — 与内置 round 同名, 模块内导出
    """逐元素四舍五入 (half away from zero, 与 torch.round 一致)"""
    return _l.round(x)


__all__ = [
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "sigmoid",
    "tanh",
    "sin",
    "cos",
    "rsqrt",
    "square",
    "reciprocal",
    "sign",
    "floor",
    "ceil",
    "round",
]
