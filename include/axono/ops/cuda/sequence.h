#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

// 语义同 cpu/sequence.h; 兼容 Graph 捕获 (当前流 + IsCapturing 守卫)。

AXONO_EXPORT core::Status Embedding(const core::Context &ctx,
                                    const core::Tensor &ids,
                                    const core::Tensor &table,
                                    core::Tensor &result);
AXONO_EXPORT core::Status Rope(const core::Context &ctx,
                               const core::Tensor &x,
                               const core::Tensor &pos_ids, float theta,
                               core::Tensor &result);
AXONO_EXPORT core::Status ScaledDotProductAttention(
    const core::Context &ctx, const core::Tensor &q, const core::Tensor &k,
    const core::Tensor &v, bool is_causal, core::Tensor &result);
AXONO_EXPORT core::Status Concat(const core::Context &ctx,
                                 const core::Tensor &a, const core::Tensor &b,
                                 int axis, core::Tensor &result);
AXONO_EXPORT core::Status Slice(const core::Context &ctx,
                                const core::Tensor &x, size_t axis,
                                size_t start, size_t length,
                                core::Tensor &result);
AXONO_EXPORT core::Status ArgmaxLastDim(const core::Context &ctx,
                                        const core::Tensor &x,
                                        core::Tensor &result);

}  // namespace cuda
}  // namespace ops
}  // namespace axono
