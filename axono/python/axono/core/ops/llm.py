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
"""LLM 相关算子: softmax/log_softmax/gelu/silu/layer_norm/rms_norm。

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


def embedding(ids, table):
    """查表 embedding: ``table[ids]``, ids 任意形状 (INT64)"""
    return _l.embedding(ids, table)


def rope(x, pos_ids, theta: float = 10000.0):
    """旋转位置编码: x (seq, heads, head_dim) 按 pos_ids 旋转 (FLOAT32)"""
    return _l.rope(x, pos_ids, theta)


def scaled_dot_product_attention(q, k, v, is_causal: bool = False):
    """缩放点积注意力 (GQA): softmax(QK^T/sqrt(d)+mask) @ V

    q: (seq, n_q_heads, head_dim); k/v: (kv_seq, n_kv_heads, head_dim)
    """
    return _l.scaled_dot_product_attention(q, k, v, is_causal)


def concat(a, b, axis: int):
    """沿 axis 拼接两个同 dtype 张量"""
    return _l.concat(a, b, axis)


def slice_(x, axis: int, start: int, length: int):
    """沿 axis 取 [start, start+length) 连续切片"""
    if axis < 0:
        axis = x.ndim + axis
    return _l.slice(x, axis, start, length)


def argmax(x):
    """沿最后一维 argmax, 返回 INT64"""
    return _l.argmax(x)
