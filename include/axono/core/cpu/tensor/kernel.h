// Axono/include/axono/core/cpu/tensor/kernel.h
#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace core {
namespace cpu {
namespace tensor {

AXONO_EXPORT core::Status DispatchFill(core::Tensor &tensor, void *value,
                                       size_t value_size);
AXONO_EXPORT core::Status DispatchZero(Tensor &tensor);  // 改成非内联

AXONO_EXPORT void TensorCopyKernel(void *dst, const void *src,
                                   size_t num_bytes);

// 模板声明（实现移到cpp，但必须显式实例化）
template <typename T>
AXONO_EXPORT void TensorZeroKernel(T *data, size_t num_elements);

}  // namespace tensor
}  // namespace cpu
}  // namespace core
}  // namespace axono
