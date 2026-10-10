#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

// 单算子融合 Linear: result = x @ W^T + bias
//   x: (m, k)  W: (out_f, in_f=k)  bias: (out_f,)  result: (m, out_f)
// bias 可为空张量 (num_elements()==0) 表示无偏置。
// 仅浮点 (FLOAT32/FLOAT64); gemm 优先 cuBLASLt, 回退经典 cuBLAS,
// beta=1 配合 bias 广播 kernel 完成融合。
// 兼容 Graph 捕获 (当前流提交 + IsCapturing 守卫)。
AXONO_EXPORT core::Status Linear(const core::Context &ctx,
                                 const core::Tensor &x,
                                 const core::Tensor &weight,
                                 const core::Tensor &bias,
                                 core::Tensor &result);

}  // namespace cuda
}  // namespace ops
}  // namespace axono
