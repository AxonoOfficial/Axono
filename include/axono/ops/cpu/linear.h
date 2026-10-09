#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cpu {

// 单算子融合 Linear: result = x @ W^T + bias
//   x: (m, k)  W: (out_f, in_f=k)  bias: (out_f,)  result: (m, out_f)
// bias 可为空张量 (num_elements()==0) 表示无偏置。
// 仅浮点 (FLOAT32/FLOAT64); BLAS 可用时走 cblas gemm beta=1,
// 否则退化为分块朴素实现。
AXONO_EXPORT core::Status Linear(const core::Context &ctx,
                                 const core::Tensor &x,
                                 const core::Tensor &weight,
                                 const core::Tensor &bias,
                                 core::Tensor &result);

}  // namespace cpu
}  // namespace ops
}  // namespace axono
