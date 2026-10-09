#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

// ---- 二元逐元素算子 (同形) ----
AXONO_EXPORT core::Status Sub(const core::Context &ctx, const core::Tensor &a,
                              const core::Tensor &b, core::Tensor &result);
AXONO_EXPORT core::Status Mul(const core::Context &ctx, const core::Tensor &a,
                              const core::Tensor &b, core::Tensor &result);
AXONO_EXPORT core::Status Div(const core::Context &ctx, const core::Tensor &a,
                              const core::Tensor &b, core::Tensor &result);

// ---- 一元逐元素算子 ----
AXONO_EXPORT core::Status Neg(const core::Context &ctx, const core::Tensor &x,
                              core::Tensor &result);
AXONO_EXPORT core::Status Abs(const core::Context &ctx, const core::Tensor &x,
                              core::Tensor &result);
AXONO_EXPORT core::Status Exp(const core::Context &ctx, const core::Tensor &x,
                              core::Tensor &result);
AXONO_EXPORT core::Status Log(const core::Context &ctx, const core::Tensor &x,
                              core::Tensor &result);
AXONO_EXPORT core::Status Sqrt(const core::Context &ctx, const core::Tensor &x,
                               core::Tensor &result);
AXONO_EXPORT core::Status Sigmoid(const core::Context &ctx,
                                  const core::Tensor &x, core::Tensor &result);
AXONO_EXPORT core::Status Tanh(const core::Context &ctx, const core::Tensor &x,
                               core::Tensor &result);

}  // namespace cuda
}  // namespace ops
}  // namespace axono
