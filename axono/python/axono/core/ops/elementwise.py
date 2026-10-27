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
    """逐元素取负 -x。

    示例::

        y = axono.neg(x)                       # 或 -x
    """
    return _l.neg(x)


def abs(x):  # noqa: A001 — 与 torch.abs 同名
    """逐元素绝对值 |x|。

    示例::

        y = axono.abs(x)                       # 或 abs(x)
    """
    return _l.abs(x)


def exp(x):
    """逐元素自然指数 e^x。

    示例::

        e = axono.exp(x)                       # softmax 手工实现的一步
    """
    return _l.exp(x)


def log(x):
    """逐元素自然对数 ln(x) (x<=0 时为 -inf/nan, 与 torch 一致)。

    示例::

        eps = Tensor.full(p.shape, 1e-12, dtype=p.dtype, device=p.device)
        l = axono.log(p + eps)
    """
    return _l.log(x)


def sqrt(x):
    """逐元素平方根。

    示例::

        eps = Tensor.full(x.shape, 1e-12, dtype=x.dtype, device=x.device)
        s = axono.sqrt(x + eps)
    """
    return _l.sqrt(x)


def sigmoid(x):
    """Sigmoid 激活 1/(1+e^-x)。

    示例::

        p = axono.sigmoid(logits)              # 二分类概率
    """
    return _l.sigmoid(x)


def tanh(x):
    """双曲正切 tanh(x)。

    示例::

        g = axono.tanh(x)                      # GRU/经典 RNN 门控
    """
    return _l.tanh(x)


def sin(x):
    """逐元素正弦。

    示例::

        pos = axono.sin(freqs * positions)     # 经典正弦位置编码
    """
    return _l.sin(x)


def cos(x):
    """逐元素余弦。

    示例::

        pos = axono.cos(freqs * positions)
    """
    return _l.cos(x)


def rsqrt(x):
    """逐元素平方根倒数 1/sqrt(x)。

    示例::

        eps = Tensor.full(mean_sq.shape, 1e-5, dtype=mean_sq.dtype, device=mean_sq.device)
        inv = axono.rsqrt(mean_sq + eps)
    """
    return _l.rsqrt(x)


def square(x):
    """逐元素平方 x*x。

    示例::

        ms = axono.square(x)                   # mean(x^2) 的一步
    """
    return _l.square(x)


def reciprocal(x):
    """逐元素倒数 1/x (x=0 处为 inf, 与 IEEE 一致)。

    示例::

        inv = axono.reciprocal(scale)
    """
    return _l.reciprocal(x)


def sign(x):
    """逐元素符号 (-1 / 0 / 1)。

    示例::

        s = axono.sign(x)
    """
    return _l.sign(x)


def floor(x):
    """逐元素向下取整。

    示例::

        n = axono.floor(x)
    """
    return _l.floor(x)


def ceil(x):
    """逐元素向上取整。

    示例::

        n = axono.ceil(x)
    """
    return _l.ceil(x)


def round(x):  # noqa: A001 — 与内置 round 同名, 模块内导出
    """逐元素取最近整数, 恰好一半时远离零 (half away from zero)。

    示例::

        n = axono.round(x)
    """
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
