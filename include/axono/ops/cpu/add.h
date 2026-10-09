#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cpu {

AXONO_EXPORT core::Status Add(const core::Context &ctx, const core::Tensor &a,
                              const core::Tensor &b, core::Tensor &result);

AXONO_EXPORT core::Status AddScalar(const core::Context &ctx,
                                    const core::Tensor &a, void *scalar,
                                    size_t scalar_size, core::Tensor &result);

}  // namespace cpu
}  // namespace ops
}  // namespace axono
