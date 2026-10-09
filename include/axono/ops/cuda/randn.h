#pragma once

#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

core::Status Randn(const core::Context& ctx, core::Tensor& out,
                   float mean = 0.0f, float stddev = 1.0f);

}  // namespace cuda
}  // namespace ops
}  // namespace axono
