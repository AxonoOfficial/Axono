#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

// CUDA 逐元素加法内核
template <typename T>
__global__ void AddKernel(const T *a, const T *b, T *result, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    result[idx] = a[idx] + b[idx];
  }
}

// CUDA 标量加法内核
template <typename T>
__global__ void AddScalarKernel(const T *a, T scalar, T *result, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    result[idx] = a[idx] + scalar;
  }
}

// 启动配置计算
inline dim3 CalculateLaunchConfig(size_t num_elements) {
  const size_t block_size = 256;
  const size_t grid_size = (num_elements + block_size - 1) / block_size;
  return dim3(grid_size);
}

// 类型分派的加法
core::Status DispatchAdd(const core::Tensor &a, const core::Tensor &b, core::Tensor &result) {
  auto num_elements = a.num_elements();

  // 检查形状一致性
  if (!a.IsSameShape(b) || !a.IsSameShape(result)) {
    return core::Status::SHAPE_MISMATCH;
  }

  // 检查数据类型一致性
  if (a.dtype() != b.dtype() || a.dtype() != result.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  auto launch_config = CalculateLaunchConfig(num_elements);
  
  // 根据数据类型选择内核
  switch (a.dtype()) {
  case core::DataType::FLOAT32:
    AddKernel<float><<<launch_config, 256>>>(
        a.data<float>(), b.data<float>(), result.data<float>(), num_elements);
    break;
  case core::DataType::FLOAT64:
    AddKernel<double><<<launch_config, 256>>>(
        a.data<double>(), b.data<double>(), result.data<double>(), num_elements);
    break;
  case core::DataType::INT32:
    AddKernel<int32_t><<<launch_config, 256>>>(
        a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), num_elements);
    break;
  case core::DataType::INT64:
    AddKernel<int64_t><<<launch_config, 256>>>(
        a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), num_elements);
    break;
  default:
    return core::Status::UNSUPPORTED_TYPE;
  }

  cudaDeviceSynchronize();
  return core::Status::OK;
}
core::Status Add(const core::Context &ctx, const core::Tensor &a, const core::Tensor &b,
           core::Tensor &result) {
  (void)ctx; // 暂时未使用

  // 基本参数检查
  if (a.ndim() != b.ndim()) {
    return core::Status::SHAPE_MISMATCH;
  }

  // 检查形状一致性
  if (!a.IsSameShape(b)) {
    return core::Status::SHAPE_MISMATCH;
  }

  // 检查数据类型一致性
  if (a.dtype() != b.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 设置结果张量的形状
  core::Status status = result.Resize(a.shape());
  if (status != core::Status::OK) {
    return status;
  }

  // 设置结果的数据类型
  if (result.dtype() != a.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 调用内核执行加法
  return DispatchAdd(a, b, result);
}

// 类型分派的标量加法
core::Status DispatchAddScalar(const core::Tensor &a, void *scalar,
                                            size_t scalar_size, core::Tensor &result) {
  auto num_elements = a.num_elements();

  // 检查形状一致性
  if (!a.IsSameShape(result)) {
    return core::Status::SHAPE_MISMATCH;
  }

  // 检查数据类型一致性
  if (a.dtype() != result.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  auto launch_config = CalculateLaunchConfig(num_elements);
  
  // 根据数据类型选择内核
  switch (a.dtype()) {
  case core::DataType::FLOAT32: {
    float scalar_value = 0.0f;
    if (scalar_size >= sizeof(float)) {
      memcpy(&scalar_value, scalar, sizeof(float));
    }
    AddScalarKernel<float><<<launch_config, 256>>>(
        a.data<float>(), scalar_value, result.data<float>(), num_elements);
    break;
  }
  case core::DataType::FLOAT64: {
    double scalar_value = 0.0;
    if (scalar_size >= sizeof(double)) {
      memcpy(&scalar_value, scalar, sizeof(double));
    }
    AddScalarKernel<double><<<launch_config, 256>>>(
        a.data<double>(), scalar_value, result.data<double>(), num_elements);
    break;
  }
  case core::DataType::INT32: {
    int32_t scalar_value = 0;
    if (scalar_size >= sizeof(int32_t)) {
      memcpy(&scalar_value, scalar, sizeof(int32_t));
    }
    AddScalarKernel<int32_t><<<launch_config, 256>>>(
        a.data<int32_t>(), scalar_value, result.data<int32_t>(), num_elements);
    break;
  }
  default:
    return core::Status::UNSUPPORTED_TYPE;
  }

  cudaDeviceSynchronize();
  return core::Status::OK;
}

core::Status AddScalar(const core::Context &ctx, const core::Tensor &a, 
                       void *scalar, size_t scalar_size, core::Tensor &result) {
  (void)ctx;

  // 检查形状和数据类型
  if (!a.IsSameShape(result)) {
    return core::Status::SHAPE_MISMATCH;
  }
  if (a.dtype() != result.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  return DispatchAddScalar(a, scalar, scalar_size, result);
}

} // namespace cuda
} // namespace ops
} // namespace axono
