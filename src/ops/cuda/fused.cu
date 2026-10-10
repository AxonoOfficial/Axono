// 融合算子 (CUDA): 多个逐元素/归约操作合并为单 kernel,
// 省去中间张量的完整读写轮次 (显存带宽是大模型推理的主要瓶颈)。
#include "axono/ops/cuda/fused.h"

#include <cuda_runtime.h>

#include <cmath>

#include "axono/core/cuda/capture.h"
#include "axono/core/cuda/stream.h"
#include "axono/core/tensor.h"

namespace axono {
namespace ops {
namespace cuda {
namespace {

constexpr int kBlock = 256;

// 当前流 (捕获时为捕获流)
inline cudaStream_t Stream() { return core::cuda::AxonoCurrentStream(); }
core::Status Finish() {
  return core::cuda::MaybeSync() == cudaSuccess ? core::Status::OK
                                                : core::Status::DEVICE_ERROR;
}

// y = silu(gate) * up       (SwiGLU 中段)
__global__ void SiluMulKernel(const float *gate, const float *up, float *y,
                              size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = gate[i];
  y[i] = g / (1.0f + expf(-g)) * up[i];
}

// y = x + residual; rms = rsqrt(mean(y^2) + eps); out = y * rms * weight
// (残差加法 + RMSNorm 融合; 每行一个 block, 行内归约)
__global__ void AddRmsNormKernel(const float *x, const float *residual,
                                 const float *weight, float *y, float *out,
                                 size_t hidden, float eps) {
  const size_t row = blockIdx.x;
  const float *xr = x + row * hidden;
  const float *rr = residual + row * hidden;
  float *yr = y + row * hidden;
  float *orr = out + row * hidden;

  float local = 0.0f;
  for (size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = xr[i] + rr[i];
    yr[i] = v;
    local += v * v;
  }
  // block 内 sum 归约 (warp shuffle)
  for (int off = 16; off > 0; off >>= 1)
    local += __shfl_down_sync(0xffffffff, local, off);
  __shared__ float warp[32];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) warp[wid] = local;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  if (threadIdx.x < 32) {
    float v = threadIdx.x < nw ? warp[threadIdx.x] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffff, v, off);
    if (threadIdx.x == 0) warp[0] = v;
  }
  __syncthreads();
  const float rms = rsqrtf(warp[0] / static_cast<float>(hidden) + eps);
  for (size_t i = threadIdx.x; i < hidden; i += blockDim.x)
    orr[i] = yr[i] * rms * weight[i];
}

}  // namespace

core::Status SiluMul(const core::Context &ctx, const core::Tensor &gate,
                     const core::Tensor &up, core::Tensor &result) {
  (void)ctx;
  if (gate.dtype() != core::DataType::FLOAT32 ||
      gate.shape() != up.shape() || gate.device() != up.device())
    return core::Status::SHAPE_MISMATCH;
  core::Status st = result.Resize(gate.shape());
  if (st != core::Status::OK) return st;
  const size_t n = gate.num_elements();
  SiluMulKernel<<<static_cast<unsigned>((n + kBlock - 1) / kBlock), kBlock, 0,
                  Stream()>>>(gate.data<float>(), up.data<float>(),
                              result.data<float>(), n);
  return Finish();
}

core::Status AddRmsNorm(const core::Context &ctx, const core::Tensor &x,
                        const core::Tensor &residual, const core::Tensor &weight,
                        float eps, core::Tensor &y, core::Tensor &out) {
  (void)ctx;
  if (x.dtype() != core::DataType::FLOAT32 || x.shape() != residual.shape())
    return core::Status::SHAPE_MISMATCH;
  if (x.ndim() < 1) return core::Status::INVALID_ARGUMENT;
  const size_t hidden = x.shape().back();
  const size_t rows = x.num_elements() / hidden;
  if (weight.ndim() != 1 || weight.shape()[0] != hidden)
    return core::Status::SHAPE_MISMATCH;
  core::Status st = y.Resize(x.shape());
  if (st != core::Status::OK) return st;
  st = out.Resize(x.shape());
  if (st != core::Status::OK) return st;
  AddRmsNormKernel<<<static_cast<unsigned>(rows), kBlock, 0, Stream()>>>(
      x.data<float>(), residual.data<float>(), weight.data<float>(),
      y.data<float>(), out.data<float>(), hidden, eps);
  return Finish();
}

}  // namespace cuda
}  // namespace ops
}  // namespace axono
