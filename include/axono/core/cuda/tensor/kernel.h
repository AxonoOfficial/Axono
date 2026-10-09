// Axono/include/axono/core/cuda/tensor/kernel.h
#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace core {
namespace cuda {
namespace tensor {
// tensor操作
AXONO_EXPORT core::Status DispatchFill(core::Tensor &tensor, void *value,
                                       size_t value_size);
AXONO_EXPORT core::Status DispatchZero(core::Tensor &tensor);

AXONO_EXPORT void TensorCopyKernel(void *dst, const void *src,
                                   size_t num_bytes);
// 模板函数声明
template <typename T>
AXONO_EXPORT core::Status TensorReadKernel(const T *device_data, T *host_data,
                                           size_t num_elements);
}  // namespace tensor

}  // namespace cuda
}  // namespace core
}  // namespace axono
