"""axono.gqa — GQA (Grouped-Query Attention) 优化算子。

decode (q_len=1) 专用 split-K attention (flash-decoding 方案):
K/V 每个 kv-head 只从显存读一次 (group 内 q-head 共享), kv 分片并行 +
online-softmax 合并。上下文越长加速越明显 (通用 SDPA 在 decode 时
加权 V 串行循环, 长 skv 是单 block 瓶颈)。
"""

import libaxono as _l

__all__ = ["gqa_decode_attention"]


def gqa_decode_attention(q, k, v):
    """GQA decode attention (q_len=1, split-K flash-decoding)。

    decode 专用: K/V 每个 kv-head 只从显存读一次 (group 内 q-head
    共享), kv 分片并行 + online-softmax 合并。长上下文 1.5-2.0x;
    短上下文自动回退通用路径。

    参数:
        q: (n_q_heads, head_dim) 当前步 query
        k_cache / v_cache: (n_kv_heads, seq_len, head_dim) 历史 KV
        scale: 1/sqrt(head_dim) 注意力缩放

    返回:
        Tensor: (n_q_heads, head_dim) 注意力输出

    示例::

        import axono.gqa as gqa
        out = gqa.gqa_decode_attention(q, k_cache, v_cache, scale)

    # 形状: q (1, hq, d); k/v (skv, hkv, d), hq % hkv == 0; 返回 (1, hq, d)
    # 数值等价于 scaled_dot_product_attention(q, k, v, is_causal=False);
    # 上下文 skv < 2048 时内部自动回退通用 SDPA (split-K 收益不抵开销)。
    """
    return _l.gqa_decode_attention(q, k, v)
