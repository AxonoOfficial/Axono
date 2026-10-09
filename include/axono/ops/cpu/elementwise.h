#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cpu {

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
AXONO_EXPORT core::Status Sin(const core::Context &ctx, const core::Tensor &x,
                              core::Tensor &result);
AXONO_EXPORT core::Status Cos(const core::Context &ctx, const core::Tensor &x,
                              core::Tensor &result);
AXONO_EXPORT core::Status Rsqrt(const core::Context &ctx, const core::Tensor &x,
                                core::Tensor &result);
AXONO_EXPORT core::Status Square(const core::Context &ctx,
                                 const core::Tensor &x, core::Tensor &result);
AXONO_EXPORT core::Status Reciprocal(const core::Context &ctx,
                                     const core::Tensor &x,
                                     core::Tensor &result);
AXONO_EXPORT core::Status Sign(const core::Context &ctx, const core::Tensor &x,
                               core::Tensor &result);
AXONO_EXPORT core::Status Floor(const core::Context &ctx, const core::Tensor &x,
                                core::Tensor &result);
AXONO_EXPORT core::Status Ceil(const core::Context &ctx, const core::Tensor &x,
                               core::Tensor &result);
AXONO_EXPORT core::Status Round(const core::Context &ctx, const core::Tensor &x,
                                core::Tensor &result);

// ---- 其它二元 ----
AXONO_EXPORT core::Status Pow(const core::Context &ctx, const core::Tensor &a,
                              const core::Tensor &b, core::Tensor &result);
AXONO_EXPORT core::Status Maximum(const core::Context &ctx,
                                  const core::Tensor &a, const core::Tensor &b,
                                  core::Tensor &result);
AXONO_EXPORT core::Status Minimum(const core::Context &ctx,
                                  const core::Tensor &a, const core::Tensor &b,
                                  core::Tensor &result);

}  // namespace cpu
}  // namespace ops
}  // namespace axono
