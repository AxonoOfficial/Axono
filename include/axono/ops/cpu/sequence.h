#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cpu {

// 查表 embedding: out (..., hidden) = table[ids]
// ids: 整型 (INT32/INT64), 任意形状; table: (num_emb, hidden) FLOAT32/FLOAT64。
AXONO_EXPORT core::Status Embedding(const core::Context &ctx,
                                    const core::Tensor &ids,
                                    const core::Tensor &table,
                                    core::Tensor &result);

// 旋转位置编码 (RoPE, 逐位置标准式): x (rows, n_heads, head_dim) 就地按
// 最后一维成对旋转 (偶, 奇), freq = theta^(-2i/head_dim), pos_ids (rows,)。
// 只支持 FLOAT32。返回输出张量 (可 = 输入实现原地语义由调用方控制)。
AXONO_EXPORT core::Status Rope(const core::Context &ctx,
                               const core::Tensor &x,
                               const core::Tensor &pos_ids, float theta,
                               core::Tensor &result);

// RoPE 变体 1: 直接用 (cos, sin) 旋转 (M-RoPE / 3D 位置)。
//   x: (seq, n_heads, head_dim) FLOAT32; cos/sin: (seq, head_dim) (广播到 heads)。
//   旋转按 HF rotate_half 约定: out[i] = x[i]*cos[i] - x[i+half]*sin[i],
//   out[i+half] = x[i+half]*cos[i+half] + x[i]*sin[i+half] (half=dim/2)。
AXONO_EXPORT core::Status RopeWithCosSin(const core::Context &ctx,
                                         const core::Tensor &x,
                                         const core::Tensor &cos,
                                         const core::Tensor &sin,
                                         core::Tensor &result);

// RoPE 变体 2: 3D 位置 (T/H/W) 走交错 (interleaved stride-3) 频率重组,
// 等价于 HF Qwen3VLTextRotaryEmbedding。pos: (3, seq) INT64, inv_freq: (dim/2,)
// FLOAT32 (= theta^(-2i/dim), i in [0,dim/2)), mrope_section: [t,h,w]。
AXONO_EXPORT core::Status RopeThd(const core::Context &ctx,
                                  const core::Tensor &x,
                                  const core::Tensor &pos,
                                  const core::Tensor &inv_freq,
                                  int t_sec, int h_sec, int w_sec,
                                  core::Tensor &result);

// 因果自注意力 (GQA): 一次算完 softmax(QK^T/sqrt(d) + mask) @ V。
//   q: (seq, n_q_heads, head_dim)   k/v: (kv_seq, n_kv_heads, head_dim)
//   n_q_heads 必须是 n_kv_heads 的整数倍 (GQA 分组广播);
//   is_causal: q 长度 1 (decode) 或 kv_seq == seq 时启用下三角因果掩码;
//   仅 FLOAT32。result: (seq, n_q_heads, head_dim)。
AXONO_EXPORT core::Status ScaledDotProductAttention(
    const core::Context &ctx, const core::Tensor &q, const core::Tensor &k,
    const core::Tensor &v, bool is_causal, core::Tensor &result);

// 沿 axis 拼接 (当前支持 axis=0/-1, 输入同 dtype)。
AXONO_EXPORT core::Status Concat(const core::Context &ctx,
                                 const core::Tensor &a, const core::Tensor &b,
                                 int axis, core::Tensor &result);

// 沿 axis 取切片 [start, start+length) (负索引不支持, contiguous)。
AXONO_EXPORT core::Status Slice(const core::Context &ctx,
                                const core::Tensor &x, size_t axis,
                                size_t start, size_t length,
                                core::Tensor &result);

// 沿最后一维 argmax, 返回 INT64 (...,)。
AXONO_EXPORT core::Status ArgmaxLastDim(const core::Context &ctx,
                                        const core::Tensor &x,
                                        core::Tensor &result);

}  // namespace cpu
}  // namespace ops
}  // namespace axono
