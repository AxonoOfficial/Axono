// GQA Attention (CUDA) 声明: decode (q_len=1) 专用 split-K 优化。
#ifndef AXONO_OPS_CUDA_GQA_H_
#define AXONO_OPS_CUDA_GQA_H_

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

// GQA decode attention: q (1, hq, d), k/v (skv, hkv, d), hq % hkv == 0。
// split-K flash-decoding: K/V 每 kv-head 只读一次, kv 分片并行 + online-softmax
// 合并。结果 (1, hq, d)。
AXONO_EXPORT core::Status GqaDecodeAttention(const core::Context &ctx,
                                             const core::Tensor &q,
                                             const core::Tensor &k,
                                             const core::Tensor &v,
                                             core::Tensor &result);

}  // namespace cuda
}  // namespace ops
}  // namespace axono

#endif  // AXONO_OPS_CUDA_GQA_H_
