// LLM 相关算子 (CUDA): softmax/log_softmax/gelu/silu/layernorm/rmsnorm。
// 归约类: 每 block 处理一行 (block 内 warp shuffle 归约, 行长分片循环);
// 逐元素类: 平铺 kernel。兼容 Graph 捕获 (当前流 + IsCapturing 守卫)。
#include <cuda_runtime.h>
#include <cstddef>

#include "axono/core/cuda/capture.h"
#include "axono/core/cuda/stream.h"
#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include "axono/ops/cuda/llm.h"

namespace axono {
namespace ops {
namespace cuda {

namespace {

constexpr int kBlock = 256;

inline cudaStream_t Stream() { return core::cuda::AxonoCurrentStream(); }

inline core::Status Finish() {
  if (cudaGetLastError() != cudaSuccess) return core::Status::DEVICE_ERROR;
  if (core::cuda::IsCapturing()) return core::Status::OK;
  return cudaDeviceSynchronize() == cudaSuccess ? core::Status::OK
                                                : core::Status::INTERNAL_ERROR;
}

// ---- block 内 max / sum 归约 ----
template <typename T>
__device__ inline T BlockReduceMax(T v) {
  // warp 内 shuffle
  for (int off = 16; off > 0; off >>= 1) {
    const T o = __shfl_down_sync(0xffffffff, v, off);
    if (o > v) v = o;
  }
  __shared__ T warp[32];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) warp[wid] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  if (threadIdx.x < 32) {
    v = threadIdx.x < nw ? warp[threadIdx.x] : T(-3.4e38);
    for (int off = 16; off > 0; off >>= 1) {
      const T o = __shfl_down_sync(0xffffffff, v, off);
      if (o > v) v = o;
    }
    if (threadIdx.x == 0) warp[0] = v;
  }
  __syncthreads();
  return warp[0];
}

template <typename T>
__device__ inline T BlockReduceSum(T v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  __shared__ T warp[32];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) warp[wid] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  if (threadIdx.x < 32) {
    v = threadIdx.x < nw ? warp[threadIdx.x] : T(0);
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffff, v, off);
    if (threadIdx.x == 0) warp[0] = v;
  }
  __syncthreads();
  return warp[0];
}

// ---- softmax / log_softmax ----
template <typename T, bool LogMode>
__global__ void SoftmaxRowKernel(const T *x, T *out, size_t rows,
                                 size_t cols) {
  const size_t r = blockIdx.x;
  if (r >= rows) return;
  const T *xr = x + r * cols;
  T *orr = out + r * cols;

  T local_max = T(-3.4e38);
  for (size_t c = threadIdx.x; c < cols; c += blockDim.x)
    local_max = xr[c] > local_max ? xr[c] : local_max;
  const T mx = BlockReduceMax<T>(local_max);

  T local_sum = T(0);
  for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const T e = exp(xr[c] - mx);
    orr[c] = e;
    local_sum += e;
  }
  const T sum = BlockReduceSum<T>(local_sum);

  if (LogMode) {
    const T lse = log(sum) + mx;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x)
      orr[c] = xr[c] - lse;
  } else {
    const T inv = T(1) / sum;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) orr[c] *= inv;
  }
}

// ---- layernorm ----
template <typename T>
__global__ void LayerNormRowKernel(const T *x, T *out, size_t rows,
                                   size_t cols, const T *weight,
                                   const T *bias, int has_w, int has_b,
                                   float eps) {
  const size_t r = blockIdx.x;
  if (r >= rows) return;
  const T *xr = x + r * cols;
  T *orr = out + r * cols;

  T local_sum = T(0);
  for (size_t c = threadIdx.x; c < cols; c += blockDim.x) local_sum += xr[c];
  const T mean = BlockReduceSum<T>(local_sum) / static_cast<T>(cols);

  T local_var = T(0);
  for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const T d = xr[c] - mean;
    local_var += d * d;
  }
  const T var = BlockReduceSum<T>(local_var) / static_cast<T>(cols);
  const T inv = rsqrt(var + static_cast<T>(eps));

  for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const T normed = (xr[c] - mean) * inv;
    orr[c] = (has_w ? normed * weight[c] : normed) +
             (has_b ? bias[c] : T(0));
  }
}

// ---- rmsnorm ----
template <typename T>
__global__ void RmsNormRowKernel(const T *x, T *out, size_t rows,
                                 size_t cols, const T *weight, int has_w,
                                 float eps) {
  const size_t r = blockIdx.x;
  if (r >= rows) return;
  const T *xr = x + r * cols;
  T *orr = out + r * cols;

  T local_ss = T(0);
  for (size_t c = threadIdx.x; c < cols; c += blockDim.x)
    local_ss += xr[c] * xr[c];
  const T ss = BlockReduceSum<T>(local_ss) / static_cast<T>(cols);
  const T inv = rsqrt(ss + static_cast<T>(eps));

  for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const T normed = xr[c] * inv;
    orr[c] = has_w ? normed * weight[c] : normed;
  }
}

// ---- gelu / silu ----
__global__ void GeluKernel(const float *x, float *out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118654752f));
}

__global__ void GeluKernelD(const double *x, double *out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = 0.5 * x[i] * (1.0 + erf(x[i] * 0.70710678118654752440));
}

__global__ void SiluKernel(const float *x, float *out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] / (1.0f + expf(-x[i]));
}

__global__ void SiluKernelD(const double *x, double *out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] / (1.0 + exp(-x[i]));
}

}  // namespace

core::Status Softmax(const core::Context &ctx, const core::Tensor &x,
                     core::Tensor &result) {
  (void)ctx;
  if (x.ndim() < 1 || x.num_elements() == 0)
    return core::Status::INVALID_ARGUMENT;
  const size_t cols = x.shape().back();
  const size_t rows = x.num_elements() / cols;
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;
  if (result.dtype() != x.dtype()) return core::Status::UNSUPPORTED_TYPE;
  if (x.dtype() == core::DataType::FLOAT32)
    SoftmaxRowKernel<float, false><<<static_cast<unsigned>(rows), kBlock, 0,
                                     Stream()>>>(
        x.data<float>(), result.data<float>(), rows, cols);
  else if (x.dtype() == core::DataType::FLOAT64)
    SoftmaxRowKernel<double, false><<<static_cast<unsigned>(rows), kBlock, 0,
                                      Stream()>>>(
        x.data<double>(), result.data<double>(), rows, cols);
  else
    return core::Status::UNSUPPORTED_TYPE;
  return Finish();
}

core::Status LogSoftmax(const core::Context &ctx, const core::Tensor &x,
                        core::Tensor &result) {
  (void)ctx;
  if (x.ndim() < 1 || x.num_elements() == 0)
    return core::Status::INVALID_ARGUMENT;
  const size_t cols = x.shape().back();
  const size_t rows = x.num_elements() / cols;
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;
  if (result.dtype() != x.dtype()) return core::Status::UNSUPPORTED_TYPE;
  if (x.dtype() == core::DataType::FLOAT32)
    SoftmaxRowKernel<float, true><<<static_cast<unsigned>(rows), kBlock, 0,
                                    Stream()>>>(
        x.data<float>(), result.data<float>(), rows, cols);
  else if (x.dtype() == core::DataType::FLOAT64)
    SoftmaxRowKernel<double, true><<<static_cast<unsigned>(rows), kBlock, 0,
                                     Stream()>>>(
        x.data<double>(), result.data<double>(), rows, cols);
  else
    return core::Status::UNSUPPORTED_TYPE;
  return Finish();
}

core::Status Gelu(const core::Context &ctx, const core::Tensor &x,
                  core::Tensor &result) {
  (void)ctx;
  if (x.num_elements() == 0) return core::Status::INVALID_ARGUMENT;
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;
  if (result.dtype() != x.dtype()) return core::Status::UNSUPPORTED_TYPE;
  const size_t n = x.num_elements();
  const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock));
  if (x.dtype() == core::DataType::FLOAT32)
    GeluKernel<<<grid, kBlock, 0, Stream()>>>(x.data<float>(),
                                              result.data<float>(), n);
  else if (x.dtype() == core::DataType::FLOAT64)
    GeluKernelD<<<grid, kBlock, 0, Stream()>>>(x.data<double>(),
                                               result.data<double>(), n);
  else
    return core::Status::UNSUPPORTED_TYPE;
  return Finish();
}

core::Status Silu(const core::Context &ctx, const core::Tensor &x,
                  core::Tensor &result) {
  (void)ctx;
  if (x.num_elements() == 0) return core::Status::INVALID_ARGUMENT;
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;
  if (result.dtype() != x.dtype()) return core::Status::UNSUPPORTED_TYPE;
  const size_t n = x.num_elements();
  const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock));
  if (x.dtype() == core::DataType::FLOAT32)
    SiluKernel<<<grid, kBlock, 0, Stream()>>>(x.data<float>(),
                                              result.data<float>(), n);
  else if (x.dtype() == core::DataType::FLOAT64)
    SiluKernelD<<<grid, kBlock, 0, Stream()>>>(x.data<double>(),
                                               result.data<double>(), n);
  else
    return core::Status::UNSUPPORTED_TYPE;
  return Finish();
}

core::Status LayerNorm(const core::Context &ctx, const core::Tensor &x,
                       const core::Tensor &weight, const core::Tensor &bias,
                       float eps, core::Tensor &result) {
  (void)ctx;
  if (x.ndim() < 1 || x.num_elements() == 0)
    return core::Status::INVALID_ARGUMENT;
  const size_t cols = x.shape().back();
  const size_t rows = x.num_elements() / cols;
  const bool has_w = weight.num_elements() != 0;
  const bool has_b = bias.num_elements() != 0;
  if ((has_w && (weight.ndim() != 1 || weight.shape()[0] != cols ||
                 weight.dtype() != x.dtype())) ||
      (has_b && (bias.ndim() != 1 || bias.shape()[0] != cols ||
                 bias.dtype() != x.dtype())))
    return core::Status::SHAPE_MISMATCH;
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;
  if (result.dtype() != x.dtype()) return core::Status::UNSUPPORTED_TYPE;
  if (x.dtype() == core::DataType::FLOAT32)
    LayerNormRowKernel<float><<<static_cast<unsigned>(rows), kBlock, 0,
                                Stream()>>>(
        x.data<float>(), result.data<float>(), rows, cols,
        has_w ? weight.data<float>() : nullptr,
        has_b ? bias.data<float>() : nullptr, has_w ? 1 : 0, has_b ? 1 : 0,
        eps);
  else if (x.dtype() == core::DataType::FLOAT64)
    LayerNormRowKernel<double><<<static_cast<unsigned>(rows), kBlock, 0,
                                 Stream()>>>(
        x.data<double>(), result.data<double>(), rows, cols,
        has_w ? weight.data<double>() : nullptr,
        has_b ? bias.data<double>() : nullptr, has_w ? 1 : 0, has_b ? 1 : 0,
        eps);
  else
    return core::Status::UNSUPPORTED_TYPE;
  return Finish();
}

core::Status RmsNorm(const core::Context &ctx, const core::Tensor &x,
                     const core::Tensor &weight, float eps,
                     core::Tensor &result) {
  (void)ctx;
  if (x.ndim() < 1 || x.num_elements() == 0)
    return core::Status::INVALID_ARGUMENT;
  const size_t cols = x.shape().back();
  const size_t rows = x.num_elements() / cols;
  const bool has_w = weight.num_elements() != 0;
  if (has_w && (weight.ndim() != 1 || weight.shape()[0] != cols ||
                weight.dtype() != x.dtype()))
    return core::Status::SHAPE_MISMATCH;
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;
  if (result.dtype() != x.dtype()) return core::Status::UNSUPPORTED_TYPE;
  if (x.dtype() == core::DataType::FLOAT32)
    RmsNormRowKernel<float><<<static_cast<unsigned>(rows), kBlock, 0,
                              Stream()>>>(
        x.data<float>(), result.data<float>(), rows, cols,
        has_w ? weight.data<float>() : nullptr, has_w ? 1 : 0, eps);
  else if (x.dtype() == core::DataType::FLOAT64)
    RmsNormRowKernel<double><<<static_cast<unsigned>(rows), kBlock, 0,
                               Stream()>>>(
        x.data<double>(), result.data<double>(), rows, cols,
        has_w ? weight.data<double>() : nullptr, has_w ? 1 : 0, eps);
  else
    return core::Status::UNSUPPORTED_TYPE;
  return Finish();
}

}  // namespace cuda
}  // namespace ops
}  // namespace axono
