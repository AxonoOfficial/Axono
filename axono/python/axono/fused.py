"""axono.fused — 融合算子库。

将推理高频的多算子组合合并为单 kernel, 省去中间张量的整轮读写
(显存/内存带宽是大模型推理的主要瓶颈):

- silu_mul(gate, up): SwiGLU 中段 ``silu(gate) * up``, MLP 每层一次。
- add_rms_norm(x, residual, weight, eps): 残差加法 + RMSNorm 融合,
  返回 ``(y, out)`` — y 为残差和 (供下一层 residual 链复用),
  out 为归一化结果。Decoder 每层两次 (attn 前后各一)。

CPU / CUDA 双实现, 接口一致。fp32。
"""

import libaxono as _l

__all__ = ["silu_mul", "add_rms_norm"]


def silu_mul(gate, up):
    """y = silu(gate) * up (逐元素, gate/up 形状一致)。

    SwiGLU MLP 中段单 kernel 融合: 省去 silu 输出与乘法中间张量的
    两轮显存读写, CPU 2.0x / CUDA 1.6-2.4x。

    参数:
        gate / up: 输入张量, 形状一致 (如 (B, L, inter))

    返回:
        Tensor: silu(gate) * up

    示例::

        import axono.fused as F
        y = F.silu_mul(w_gate @ x, w_up @ x)   # 一步替代 silu + mul
    """
    return _l.silu_mul(gate, up)


def add_rms_norm(x, residual, weight, eps: float = 1e-6):
    """y = x + residual; out = y * rsqrt(mean(y²) + eps) * weight。

    返回 (y, out): y 是新残差基, out 是 norm 后结果。
    """
    return _l.add_rms_norm(x, residual, weight, eps)
