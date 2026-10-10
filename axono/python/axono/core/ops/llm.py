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
"""LLM 相关算子: softmax/log_softmax/gelu/gelu_tanh/silu/layer_norm/rms_norm。

全部走 libaxono 绑定 (CPU/CUDA 后端自动分派), 非原地。
"""

import libaxono as _l

from ..tensor import Tensor  # noqa: F401 — 供类型提示 (绑定类, Pyright 无法解析)


def softmax(x):
    """沿最后一维的数值稳定 softmax"""
    return _l.softmax(x)


def log_softmax(x):
    """沿最后一维的 log_softmax (= x - logsumexp(x))"""
    return _l.log_softmax(x)


def gelu(x):
    """GELU 激活 (精确 erf 式): 0.5*x*(1+erf(x/sqrt(2)))"""
    return _l.gelu(x)


def gelu_tanh(x):
    """GELU 激活 (tanh 近似式): 0.5*x*(1+tanh(sqrt(2/pi)(x+0.044715x^3)))"""
    return _l.gelu_tanh(x)


def silu(x):
    """SiLU/Swish 激活: x * sigmoid(x)"""
    return _l.silu(x)


def layer_norm(x, weight, bias, eps: float = 1e-5):
    """LayerNorm: 沿最后一维 (x-mean)/sqrt(var+eps)*weight+bias"""
    return _l.layer_norm(x, weight, bias, eps)


def rms_norm(x, weight, eps: float = 1e-5):
    """RMSNorm: x / sqrt(mean(x^2)+eps) * weight"""
    return _l.rms_norm(x, weight, eps)


def linear(x, weight, bias):
    """单算子融合 Linear: ``x @ weight.T + bias``。

    x: (m, k); weight: (out_f, k); bias: (out_f,)。
    CPU 走 cblas, CUDA 走 cuBLAS(Lt), 单次提交。
    """
    return _l.linear(x, weight, bias)


def linear_nobias(x, weight):
    """无偏置版 Linear: ``x @ weight.T``"""
    return _l.linear_nobias(x, weight)


__all__ = [
    "softmax",
    "log_softmax",
    "gelu",
    "gelu_tanh",
    "silu",
    "layer_norm",
    "rms_norm",
    "linear",
    "linear_nobias",
]
