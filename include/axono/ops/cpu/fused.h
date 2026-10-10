// 融合算子 (CPU) 声明。
#ifndef AXONO_OPS_CPU_FUSED_H_
#define AXONO_OPS_CPU_FUSED_H_

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cpu {

// y = silu(gate) * up  (SwiGLU 中段, 逐元素, 形状一致)
AXONO_EXPORT core::Status SiluMul(const core::Context &ctx,
                                  const core::Tensor &gate,
                                  const core::Tensor &up,
                                  core::Tensor &result);

// 融合残差 + RMSNorm: y = x + residual; out = y * rsqrt(mean(y²)+eps) * weight。
AXONO_EXPORT core::Status AddRmsNorm(const core::Context &ctx,
                                     const core::Tensor &x,
                                     const core::Tensor &residual,
                                     const core::Tensor &weight, float eps,
                                     core::Tensor &y, core::Tensor &out);

}  // namespace cpu
}  // namespace ops
}  // namespace axono

#endif  // AXONO_OPS_CPU_FUSED_H_
