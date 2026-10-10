#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cpu {

// ---- LLM 相关算子 (沿最后一维归约/变换, 支持任意前置维) ----
// 均仅浮点 (FLOAT32/FLOAT64)。

// 数值稳定 softmax: 减行最大值后 exp/sum。
AXONO_EXPORT core::Status Softmax(const core::Context &ctx,
                                  const core::Tensor &x,
                                  core::Tensor &result);
// log_softmax = x - log(sum(exp(x - max)))。
AXONO_EXPORT core::Status LogSoftmax(const core::Context &ctx,
                                     const core::Tensor &x,
                                     core::Tensor &result);
// GELU (精确 erf 式): 0.5*x*(1+erf(x/sqrt(2)))。
AXONO_EXPORT core::Status Gelu(const core::Context &ctx,
                               const core::Tensor &x, core::Tensor &result);
// GELU (tanh 近似式): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3)))。
AXONO_EXPORT core::Status GeluTanh(const core::Context &ctx,
                                   const core::Tensor &x,
                                   core::Tensor &result);
// SiLU (Swish): x * sigmoid(x)。
AXONO_EXPORT core::Status Silu(const core::Context &ctx,
                               const core::Tensor &x, core::Tensor &result);
// LayerNorm: 沿最后一维 (x-mean)/sqrt(var+eps)*weight+bias。
// weight/bias 形状须等于最后一维; 可传空张量表示无仿射。
AXONO_EXPORT core::Status LayerNorm(const core::Context &ctx,
                                    const core::Tensor &x,
                                    const core::Tensor &weight,
                                    const core::Tensor &bias, float eps,
                                    core::Tensor &result);
// RMSNorm: x / sqrt(mean(x^2)+eps) * weight。
// weight 形状须等于最后一维; 可传空张量表示无缩放。
AXONO_EXPORT core::Status RmsNorm(const core::Context &ctx,
                                  const core::Tensor &x,
                                  const core::Tensor &weight, float eps,
                                  core::Tensor &result);

}  // namespace cpu
}  // namespace ops
}  // namespace axono
