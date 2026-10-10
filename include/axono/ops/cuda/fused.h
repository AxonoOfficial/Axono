// 融合算子 (CUDA) 声明。
#ifndef AXONO_OPS_CUDA_FUSED_H_
#define AXONO_OPS_CUDA_FUSED_H_

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

// y = silu(gate) * up  (SwiGLU 中段, 逐元素, 形状一致)
AXONO_EXPORT core::Status SiluMul(const core::Context &ctx,
                                  const core::Tensor &gate,
                                  const core::Tensor &up,
                                  core::Tensor &result);

// 融合残差 + RMSNorm: y = x + residual; out = y * rsqrt(mean(y²)+eps) * weight。
// y 输出残差和 (供下一层复用), out 输出 norm 后结果。末维 hidden 为归约维。
AXONO_EXPORT core::Status AddRmsNorm(const core::Context &ctx,
                                     const core::Tensor &x,
                                     const core::Tensor &residual,
                                     const core::Tensor &weight, float eps,
                                     core::Tensor &y, core::Tensor &out);

}  // namespace cuda
}  // namespace ops
}  // namespace axono

#endif  // AXONO_OPS_CUDA_FUSED_H_
