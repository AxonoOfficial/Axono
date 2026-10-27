"""axono.gqa — GQA (Grouped-Query Attention) 优化算子。

CUDA decode (q_len=1) 专用 split-K attention (flash-decoding 方案):
沿 KV 序列分片并行计算，再通过 online-softmax 合并。
具体性能需在目标硬件和输入规模上测量。
"""

import libaxono as _l

__all__ = ["gqa_decode_attention"]


def gqa_decode_attention(q, k, v):
    """GQA decode attention (q_len=1, split-K flash-decoding)。

    仅支持 CUDA FLOAT32 张量。seq_len < 2048 时回退通用 SDPA。
    n_q_heads 必须能被 n_kv_heads 整除，缩放因子内部使用 1/sqrt(head_dim)。

    参数:
        q: (1, n_q_heads, head_dim) 当前步 query
        k / v: (seq_len, n_kv_heads, head_dim) 历史 KV

    返回:
        Tensor: (1, n_q_heads, head_dim) 注意力输出

    示例::

        import axono.gqa as gqa
        out = gqa.gqa_decode_attention(q, k_cache, v_cache)

    数值等价于 scaled_dot_product_attention(q, k, v, is_causal=False)。
    """
    return _l.gqa_decode_attention(q, k, v)
