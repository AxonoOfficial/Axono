#pragma once

#include <cstddef>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"

namespace axono {
namespace ops {
namespace cpu {

core::Status MatMul(const core::Context &ctx, const core::Tensor &a,
                    const core::Tensor &b, core::Tensor &result);

// 原地累加变体: result += a @ b 的 CPU 等价实现 ——
// 先把 a@b 写入 result 再叠加, 或对支持的场景直接原地累加。
// result 形状必须为 (a.shape[0], b.shape()[1])。
core::Status MatMulAccumulate(const core::Context &ctx,
                              const core::Tensor &a, const core::Tensor &b,
                              core::Tensor &result);

}  // namespace cpu
}  // namespace ops
}  // namespace axono
