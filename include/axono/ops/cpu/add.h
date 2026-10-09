#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cpu {

AXONO_EXPORT core::Status Add(const core::Context &ctx, const core::Tensor &a,
                              const core::Tensor &b, core::Tensor &result);

// 原地变体: result 可与 a/b 同一存储 (真 inplace, 无临时分配)。
// 语义: result += b (若 result 即 a) 或显式写入 a+b。
AXONO_EXPORT core::Status AddInto(const core::Context &ctx,
                                  const core::Tensor &a,
                                  const core::Tensor &b, core::Tensor &result);

AXONO_EXPORT core::Status AddScalar(const core::Context &ctx,
                                    const core::Tensor &a, void *scalar,
                                    size_t scalar_size, core::Tensor &result);

}  // namespace cpu
}  // namespace ops
}  // namespace axono
