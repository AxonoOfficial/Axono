#include <algorithm>
#include <cuda_runtime.h>
#include <cstddef>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

template <typename T>
__global__ void ReluKernel(const T* __restrict__ input,
                          T* __restrict__ output,
                          int num_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    T value = input[idx];
    output[idx] = (value > T(0)) ? value : T(0);
  }
}

template <typename T>
__global__ void ReluInplaceKernel(T* data, int num_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    T value = data[idx];
    data[idx] = (value > T(0)) ? value : T(0);
  }
}

inline dim3 CalculateLaunchConfig(int num_elements) {
  const int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;
  const int max_grid = 65535;
  if (grid_size > max_grid) grid_size = max_grid;
  if (grid_size == 0) grid_size = 1;
  return dim3(grid_size);
}

core::Status DispatchRelu(const core::Tensor& input, core::Tensor& output) {
  size_t n = input.num_elements();
  if (n > INT_MAX) return core::Status::INVALID_ARGUMENT;
  int num_el = static_cast<int>(n);

  if (!input.IsSameShape(output)) return core::Status::SHAPE_MISMATCH;
  if (input.dtype() != output.dtype()) return core::Status::UNSUPPORTED_TYPE;
  if (!input.is_cuda() || !output.is_cuda()) return core::Status::DEVICE_MISMATCH;

  dim3 grid = CalculateLaunchConfig(num_el);
  const int block = 256;
  cudaError_t err = cudaSuccess;

  switch (input.dtype()) {
    case core::DataType::FLOAT32:
      ReluKernel<float><<<grid, block>>>(input.data<float>(),
                                         output.data<float>(),
                                         num_el);
      err = cudaGetLastError();
      break;
    case core::DataType::FLOAT64:
      ReluKernel<double><<<grid, block>>>(input.data<double>(),
                                          output.data<double>(),
                                          num_el);
      err = cudaGetLastError();
      break;
    case core::DataType::INT32:
      ReluKernel<int32_t><<<grid, block>>>(input.data<int32_t>(),
                                           output.data<int32_t>(),
                                           num_el);
      err = cudaGetLastError();
      break;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }
  if (err != cudaSuccess) return core::Status::DEVICE_ERROR;

  err = cudaDeviceSynchronize();
  return (err == cudaSuccess) ? core::Status::OK : core::Status::DEVICE_ERROR;
}

core::Status DispatchReluInplace(core::Tensor& tensor) {
  size_t n = tensor.num_elements();
  if (n > INT_MAX) return core::Status::INVALID_ARGUMENT;
  int num_el = static_cast<int>(n);

  if (!tensor.is_cuda()) return core::Status::DEVICE_MISMATCH;

  dim3 grid = CalculateLaunchConfig(num_el);
  const int block = 256;
  cudaError_t err = cudaSuccess;

  switch (tensor.dtype()) {
    case core::DataType::FLOAT32:
      ReluInplaceKernel<float><<<grid, block>>>(tensor.data<float>(), num_el);
      err = cudaGetLastError();
      break;
    case core::DataType::FLOAT64:
      ReluInplaceKernel<double><<<grid, block>>>(tensor.data<double>(), num_el);
      err = cudaGetLastError();
      break;
    case core::DataType::INT32:
      ReluInplaceKernel<int32_t><<<grid, block>>>(tensor.data<int32_t>(), num_el);
      err = cudaGetLastError();
      break;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }
  if (err != cudaSuccess) return core::Status::DEVICE_ERROR;

  err = cudaDeviceSynchronize();
  return (err == cudaSuccess) ? core::Status::OK : core::Status::DEVICE_ERROR;
}

core::Status Relu(const core::Context& ctx,
                  const core::Tensor& input,
                  core::Tensor& output) {
  (void)ctx;
  auto st = output.Resize(input.shape());
  return (st == core::Status::OK) ? DispatchRelu(input, output) : st;
}

core::Status ReluInplace(const core::Context& ctx,
                         core::Tensor& tensor) {
  (void)ctx;
  return DispatchReluInplace(tensor);
}

} // namespace cuda
} // namespace ops
} // namespace axono
