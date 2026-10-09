#pragma once

#include <cstddef>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"

namespace axono {
namespace ops {
namespace cuda {

core::Status MatMul(const core::Context &ctx, const core::Tensor &a,
                    const core::Tensor &b, core::Tensor &result);

// cuBLAS 累加变体: result = result + a @ b (beta=1), 无额外分配。
// result 形状必须为 (a.shape[0], b.shape[1])。
core::Status MatMulAccumulate(const core::Context &ctx,
                              const core::Tensor &a, const core::Tensor &b,
                              core::Tensor &result);

}  // namespace cuda
}  // namespace ops
}  // namespace axono
