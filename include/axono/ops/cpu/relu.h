#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cpu {

core::Status DispatchRelu(const core::Tensor &input, core::Tensor &output);
core::Status DispatchReluInplace(core::Tensor &tensor);
core::Status Relu(const core::Context &ctx, const core::Tensor &input,
                  core::Tensor &output);
core::Status ReluInplace(const core::Context &ctx, core::Tensor &tensor);

}  // namespace cpu
}  // namespace ops
}  // namespace axono
