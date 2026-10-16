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
    """沿最后一维的数值稳定 softmax (先减 max 防溢出)。

    示例::

        logits = model(ids)                    # (B, vocab)
        probs = axono.softmax(logits)          # 每行和为 1
    """
    return _l.softmax(x)


def log_softmax(x):
    """沿最后一维的 log_softmax (= x - logsumexp(x))。

    数值上比先 softmax 再取 log 更稳定; 训练 loss 常用。

    示例::

        logp = axono.log_softmax(logits)
        nll = -logp[target]                    # 单点负对数似然
    """
    return _l.log_softmax(x)


def gelu(x):
    """GELU 激活 (精确 erf 式): 0.5*x*(1+erf(x/sqrt(2)))。

    与 torch.nn.functional.gelu 默认实现对齐。

    示例::

        h = axono.gelu(fc1(x))                 # MLP 中间层激活
    """
    return _l.gelu(x)


def gelu_tanh(x):
    """GELU 激活 (tanh 近似式): 0.5*x*(1+tanh(sqrt(2/pi)(x+0.044715x^3)))。

    torch.nn.functional.gelu(approximate="tanh") 同款; 大多数 LLM
    (LLaMA/Qwen) 配置使用近似式。

    示例::

        h = axono.gelu_tanh(fc1(x))
    """
    return _l.gelu_tanh(x)


def silu(x):
    """SiLU/Swish 激活: x * sigmoid(x)。

    SwiGLU MLP 的门控前半 (配合 axono.fused.silu_mul 一次算完)。

    示例::

        gate = axono.silu(w_gate @ x)
        up = w_up @ x
        y = w_down @ (gate * up)               # 或用 fused.silu_mul
    """
    return _l.silu(x)


def layer_norm(x, weight, bias, eps: float = 1e-5):
    """LayerNorm: 沿最后一维 (x-mean)/sqrt(var+eps) * weight + bias。

    参数:
        x: 输入 (任意 shape, 沿最后一维归一化)
        weight / bias: 一维缩放与偏移 (与最后一维等长)
        eps: 数值稳定项, 默认 1e-5

    示例::

        y = axono.layer_norm(h, weight, bias)   # Transformer 层内归一化
    """
    return _l.layer_norm(x, weight, bias, eps)


def rms_norm(x, weight, eps: float = 1e-5):
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight (无均值中心化)。

    LLaMA/Qwen 系模型的标准归一化; 推理高频算子, 融合版见
    axono.fused.add_rms_norm (残差 + 归一化单 kernel)。

    示例::

        y = axono.rms_norm(h, weight, eps=1e-6)
    """
    return _l.rms_norm(x, weight, eps)


def linear(x, weight, bias):
    """单算子融合 Linear: ``x @ weight.T + bias``。

    x: (m, k); weight: (out_f, k); bias: (out_f,)。
    CPU 走 cblas, CUDA 走 cuBLAS(Lt), 单次提交。
    """
    return _l.linear(x, weight, bias)


def linear_nobias(x, weight):
    """无偏置版 Linear: ``x @ weight.T``。

    weight 形状 (out_features, in_features), 与 nn.Linear 存储一致。

    示例::

        y = axono.linear_nobias(x, w)          # (B, K) @ w.T → (B, N)
    """
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
