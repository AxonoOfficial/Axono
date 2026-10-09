#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

// ---- LLM 相关算子 (沿最后一维归约/变换, 支持任意前置维) ----
// 均仅浮点 (FLOAT32/FLOAT64); 兼容 Graph 捕获 (当前流 + IsCapturing 守卫)。

AXONO_EXPORT core::Status Softmax(const core::Context &ctx,
                                  const core::Tensor &x,
                                  core::Tensor &result);
AXONO_EXPORT core::Status LogSoftmax(const core::Context &ctx,
                                     const core::Tensor &x,
                                     core::Tensor &result);
AXONO_EXPORT core::Status Gelu(const core::Context &ctx,
                               const core::Tensor &x, core::Tensor &result);
AXONO_EXPORT core::Status Silu(const core::Context &ctx,
                               const core::Tensor &x, core::Tensor &result);
// weight/bias 可为空张量表示无仿射。
AXONO_EXPORT core::Status LayerNorm(const core::Context &ctx,
                                    const core::Tensor &x,
                                    const core::Tensor &weight,
                                    const core::Tensor &bias, float eps,
                                    core::Tensor &result);
// weight 可为空张量表示无缩放。
AXONO_EXPORT core::Status RmsNorm(const core::Context &ctx,
                                  const core::Tensor &x,
                                  const core::Tensor &weight, float eps,
                                  core::Tensor &result);

}  // namespace cuda
}  // namespace ops
}  // namespace axono
