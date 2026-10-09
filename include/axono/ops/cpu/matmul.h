#pragma once

#include <cstddef>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"

namespace axono {
namespace ops {
namespace cpu {

core::Status MatMul(const core::Context &ctx, const core::Tensor &a,
                    const core::Tensor &b, core::Tensor &result);

}  // namespace cpu
}  // namespace ops
}  // namespace axono
