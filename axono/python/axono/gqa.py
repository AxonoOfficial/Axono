"""axono.gqa — GQA (Grouped-Query Attention) 优化算子。

decode (q_len=1) 专用 split-K attention (flash-decoding 方案):
K/V 每个 kv-head 只从显存读一次 (group 内 q-head 共享), kv 分片并行 +
online-softmax 合并。上下文越长加速越明显 (通用 SDPA 在 decode 时
加权 V 串行循环, 长 skv 是单 block 瓶颈)。
"""

import libaxono as _l

__all__ = ["gqa_decode_attention"]


def gqa_decode_attention(q, k, v):
    """GQA decode attention。

    q: (1, hq, d); k/v: (skv, hkv, d); hq % hkv == 0。返回 (1, hq, d)。
    数值等价于 scaled_dot_product_attention(q, k, v, is_causal=False)。
    上下文 skv < 2048 时内部自动回退通用 SDPA (split-K 收益不抵开销)。
    """
    return _l.gqa_decode_attention(q, k, v)
