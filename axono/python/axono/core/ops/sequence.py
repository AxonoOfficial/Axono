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
"""序列/位置算子: rope / m-rope / 注意力 等 (薄封装 libaxono)。"""

import libaxono as _l


def rope(x, pos_ids, theta: float = 10000.0):
    """旋转位置编码: x (seq, heads, head_dim) 按 pos_ids 旋转 (FLOAT32)"""
    return _l.rope(x, pos_ids, theta)


def rope_with_cos_sin(x, cos, sin):
    """用 (cos, sin) 旋转 RoPE: x (seq, heads, d); cos/sin (seq, d) 广播到 heads。

    按 HF rotate_half 约定旋转, 适合 3D/M-RoPE 等需要外部提供 cos/sin 的场景。
    """
    return _l.rope_with_cos_sin(x, cos, sin)


def rope_thd(x, pos, inv_freq, mrope_section):
    """3D (T/H/W) M-RoPE 交错频率重组: pos (3, seq) INT64, inv_freq (d/2,);

    mrope_section = [t, h, w]。等价 HF Qwen3VLTextRotaryEmbedding (stride-3 交错)。
    """
    t_sec, h_sec, w_sec = (list(mrope_section) + [0, 0, 0])[:3]
    return _l.rope_thd(x, pos, inv_freq, int(t_sec), int(h_sec), int(w_sec))


def scaled_dot_product_attention(q, k, v, is_causal: bool = False):
    """缩放点积注意力 (GQA): softmax(QK^T/sqrt(d)+mask) @ V

    q: (seq, n_q_heads, head_dim); k/v: (kv_seq, n_kv_heads, head_dim)
    """
    return _l.scaled_dot_product_attention(q, k, v, is_causal)


def embedding(ids, table):
    """查表 embedding: ``table[ids]``, ids 任意形状 (INT64)"""
    return _l.embedding(ids, table)


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


def mrope_cos_sin(pos, inv_freq, h_sec: int, w_sec: int):
    """M-RoPE cos/sin 表 (纯 Tensor, 无 numpy)。

    pos: (3, seq) INT64 (T/H/W 位置); inv_freq: (half,) FLOAT32;
    返回 (cos, sin): 各 (seq, 2*half) FLOAT32 (前后两半相同, rotate_half 配对)。
    """
    return _l.mrope_cos_sin(pos, inv_freq, h_sec, w_sec)


__all__ = [
    "rope",
    "rope_with_cos_sin",
    "rope_thd",
    "mrope_cos_sin",
    "scaled_dot_product_attention",
    "embedding",
    "concat",
    "slice_",
    "argmax",
]
