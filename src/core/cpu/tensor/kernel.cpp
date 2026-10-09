// Axono/src/core/cpu/tensor/kernel.cpp
#include "axono/core/cpu/tensor/kernel.h"

#include <cstring>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"

namespace axono {
namespace core {
namespace cpu {
namespace tensor {
// Tensor 数据填充内核
template <typename T>
AXONO_FORCE_INLINE void TensorFillKernel(T *data, size_t num_elements,
                                         T value) {
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = value;
  }
}

// Tensor 零填充内核
template <typename T>
AXONO_FORCE_INLINE void TensorZeroKernel(T *data, size_t num_elements) {
  std::memset(data, 0, num_elements * sizeof(T));
}

// Tensor 拷贝内核
AXONO_FORCE_INLINE void TensorCopyKernel(void *dst, const void *src,
                                         size_t num_bytes) {
  std::memcpy(dst, src, num_bytes);
}

// 类型分派的填充函数
core::Status DispatchFill(Tensor &tensor, void *value, size_t value_size) {
  auto num_elements = tensor.num_elements();
  auto dtype = tensor.dtype();

  switch (dtype) {
    case core::DataType::INT8: {
      int8_t fill_value = 0;
      if (value_size >= sizeof(int8_t)) {
        std::memcpy(&fill_value, value, sizeof(int8_t));
      }
      TensorFillKernel(tensor.data<int8_t>(), num_elements, fill_value);
      break;
    }
    case core::DataType::INT16: {
      int16_t fill_value = 0;
      if (value_size >= sizeof(int16_t)) {
        std::memcpy(&fill_value, value, sizeof(int16_t));
      }
      TensorFillKernel(tensor.data<int16_t>(), num_elements, fill_value);
      break;
    }
    case core::DataType::INT32: {
      int32_t fill_value = 0;
      if (value_size >= sizeof(int32_t)) {
        std::memcpy(&fill_value, value, sizeof(int32_t));
      }
      TensorFillKernel(tensor.data<int32_t>(), num_elements, fill_value);
      break;
    }
    case core::DataType::INT64: {
      int64_t fill_value = 0;
      if (value_size >= sizeof(int64_t)) {
        std::memcpy(&fill_value, value, sizeof(int64_t));
      }
      TensorFillKernel(tensor.data<int64_t>(), num_elements, fill_value);
      break;
    }
    case core::DataType::FLOAT32: {
      float fill_value = 0.0f;
      if (value_size >= sizeof(float)) {
        std::memcpy(&fill_value, value, sizeof(float));
      }
      TensorFillKernel(tensor.data<float>(), num_elements, fill_value);
      break;
    }
    case core::DataType::FLOAT64: {
      double fill_value = 0.0;
      if (value_size >= sizeof(double)) {
        std::memcpy(&fill_value, value, sizeof(double));
      }
      TensorFillKernel(tensor.data<double>(), num_elements, fill_value);
      break;
    }
    case core::DataType::BOOLEAN: {
      bool fill_value = false;
      if (value_size >= sizeof(bool)) {
        std::memcpy(&fill_value, value, sizeof(bool));
      }
      TensorFillKernel(tensor.data<bool>(), num_elements, fill_value);
      break;
    }
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }

  return core::Status::OK;
}

AXONO_FORCE_INLINE core::Status DispatchZero(Tensor &tensor) {
  return tensor.FillZero();
}

template void TensorZeroKernel<int8_t>(int8_t *, size_t);
template void TensorZeroKernel<int16_t>(int16_t *, size_t);
template void TensorZeroKernel<int32_t>(int32_t *, size_t);
template void TensorZeroKernel<int64_t>(int64_t *, size_t);
template void TensorZeroKernel<float>(float *, size_t);
template void TensorZeroKernel<double>(double *, size_t);
template void TensorZeroKernel<bool>(bool *, size_t);

}  // namespace tensor
}  // namespace cpu
}  // namespace core
}  // namespace axono
