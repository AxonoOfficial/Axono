// Axono/src/core/cuda/tensor/tensor.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

// 遵循 Lib, Type, Where, Class
namespace axono::core::cuda::tensor {

// CUDA Tensor 数据填充内核
template <typename T>
__global__ void TensorFillKernel(T *data, size_t num_elements, T value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    data[idx] = value;
  }
}

// CUDA Tensor 零填充内核
template <typename T>
__global__ void TensorZeroKernel(T *data, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    data[idx] = static_cast<T>(0);
  }
}

// 启动配置计算
inline dim3 CalculateLaunchConfig(size_t num_elements) {
  const size_t block_size = 256;
  const size_t grid_size = (num_elements + block_size - 1) / block_size;
  return dim3(grid_size);
}

// DispatchFill 实现
core::Status DispatchFill(Tensor &tensor, void *value, size_t value_size) {
  
  auto num_elements = tensor.num_elements();
  auto dtype = tensor.dtype();
  auto launch_config = CalculateLaunchConfig(num_elements);

  switch (dtype) {
  case DataType::FLOAT32: {
    float fill_value = 0.0f;
    if (value_size >= sizeof(float)) {
      std::memcpy(&fill_value, value, sizeof(float));
    }
    TensorFillKernel<float><<<launch_config, 256>>>(
        tensor.data<float>(), num_elements, fill_value);

    break;
  }
  case DataType::FLOAT64: {
    double fill_value = 0.0f;
    if (value_size >= sizeof(double)) {
      std::memcpy(&fill_value, value, sizeof(double));
    }
    TensorFillKernel<double><<<launch_config, 256>>>(
        tensor.data<double>(), num_elements, fill_value);

    break;
  }
  case DataType::INT16: {
    int16_t fill_value = 0;
    if (value_size >= sizeof(int16_t)) {
      std::memcpy(&fill_value, value, sizeof(int16_t));
    }
    TensorFillKernel<int16_t><<<launch_config, 256>>>(
        tensor.data<int16_t>(), num_elements, fill_value);

    break;
  }
  case DataType::BOOLEAN: {
    bool fill_value = false;
    if (value_size >= 1) {
      std::memcpy(&fill_value, value, 1);
    }
    TensorFillKernel<bool><<<launch_config, 256>>>(
        tensor.data<bool>(), num_elements, fill_value);
    break;
  }
  case DataType::INT8: {
    int8_t fill_value = 0;
    if (value_size >= sizeof(int8_t)) {
      std::memcpy(&fill_value, value, sizeof(int8_t));
    }
    TensorFillKernel<int8_t><<<launch_config, 256>>>(
        tensor.data<int8_t>(), num_elements, fill_value);

    break;
  }
  case DataType::INT32: {
    int32_t fill_value = 0;
    if (value_size >= sizeof(int32_t)) {
      std::memcpy(&fill_value, value, sizeof(int32_t));
    }
    TensorFillKernel<int32_t><<<launch_config, 256>>>(
        tensor.data<int32_t>(), num_elements, fill_value);
    break;
  }
  case DataType::INT64: {
    int64_t fill_value = 0;
    if (value_size >= sizeof(int64_t)) {
      std::memcpy(&fill_value, value, sizeof(int64_t));
    }
    TensorFillKernel<int64_t><<<launch_config, 256>>>(
        tensor.data<int64_t>(), num_elements, fill_value);
    break;
  }
  default:
    return core::Status::UNSUPPORTED_TYPE;
  }

  cudaDeviceSynchronize();
  return core::Status::OK;
}

// DispatchZero 实现
core::Status DispatchZero(Tensor &tensor) {
  
  auto num_elements = tensor.num_elements();
  auto dtype = tensor.dtype();
  auto launch_config = CalculateLaunchConfig(num_elements);

  switch (dtype) {
  case DataType::FLOAT32:
    TensorZeroKernel<float><<<launch_config, 256>>>(tensor.data<float>(), num_elements);
    break;
  case DataType::INT32:
    TensorZeroKernel<int32_t><<<launch_config, 256>>>(tensor.data<int32_t>(), num_elements);
    break;
  default:
    return core::Status::UNSUPPORTED_TYPE;
  }

  cudaDeviceSynchronize();
  return core::Status::OK;
}

// TensorCopyKernel 实现
void TensorCopyKernel(void *dst, const void *src, size_t num_bytes) {
    cudaMemcpy(dst, src, num_bytes, cudaMemcpyDeviceToDevice);
}

template <typename T>
core::Status TensorReadKernel(const T* device_data, T* host_data, size_t num_elements) {
    if (!device_data || !host_data || num_elements == 0) {
        return core::Status::INVALID_ARGUMENT;
    }
    cudaError_t err = cudaMemcpy(host_data, device_data, num_elements * sizeof(T), cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? core::Status::OK : core::Status::INTERNAL_ERROR;
}

// 显式实例化
template core::Status TensorReadKernel<float>(const float*, float*, size_t);
template core::Status TensorReadKernel<double>(const double*, double*, size_t);
template core::Status TensorReadKernel<int32_t>(const int32_t*, int32_t*, size_t);
template core::Status TensorReadKernel<int16_t>(const int16_t*, int16_t*, size_t);
template core::Status TensorReadKernel<int8_t>(const int8_t*, int8_t*, size_t);
template core::Status TensorReadKernel<int64_t>(const int64_t*, int64_t*, size_t);
template core::Status TensorReadKernel<bool>(const bool*, bool*, size_t);

}
