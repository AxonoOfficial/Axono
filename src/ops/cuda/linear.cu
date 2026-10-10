// 单算子融合 Linear (CUDA): result = x @ W^T + bias。
// 支持任意前置维 batch: x (..., k) -> result (..., out_f)。
// 步骤: ① bias 广播 kernel 写入 result (无 bias 则清零);
//      ② cuBLAS(Lt) gemm beta=1 累加 x @ W^T。
// 兼容 Graph 捕获: 当前流提交 + IsCapturing 守卫。
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "axono/core/cuda/capture.h"
#include "axono/core/cuda/stream.h"
#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include "axono/ops/cuda/linear.h"
#include "axono/ops/cuda/matmul_lt.h"

namespace axono {
namespace ops {
namespace cuda {

// matmul.cu 中的 cuBLAS 句柄; linear.cu 复用 (链接可见)。
cublasHandle_t GetCublasHandle();

#define AXONO_CUBLAS_CHECK(expr)                                          \
  do {                                                                    \
    cublasStatus_t s_ = (expr);                                           \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                    \
      throw std::runtime_error(std::string("cuBLAS error: ") +            \
                               std::to_string(static_cast<int>(s_)) +     \
                               " at " #expr);                             \
    }                                                                     \
  } while (0)

namespace {

__global__ void BroadcastBiasF32(float *out, const float *bias, size_t rows,
                                 size_t cols) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < rows * cols) out[idx] = bias[idx % cols];
}

__global__ void BroadcastBiasF64(double *out, const double *bias, size_t rows,
                                 size_t cols) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < rows * cols) out[idx] = bias[idx % cols];
}

__global__ void FillZeroF32(float *out, size_t n) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 0.0f;
}

__global__ void FillZeroF64(double *out, size_t n) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 0.0;
}

}  // namespace

core::Status Linear(const core::Context &ctx, const core::Tensor &x,
                    const core::Tensor &weight, const core::Tensor &bias,
                    core::Tensor &result) {
  (void)ctx;
  if (x.ndim() < 1 || weight.ndim() != 2)
    return core::Status::INVALID_ARGUMENT;
  const size_t k = x.shape().back();
  const size_t out_f = weight.shape()[0];
  const size_t rows = x.num_elements() / k;
  if (weight.shape()[1] != k) return core::Status::SHAPE_MISMATCH;
  const bool has_bias = bias.num_elements() != 0;
  if (has_bias && (bias.ndim() != 1 || bias.shape()[0] != out_f))
    return core::Status::SHAPE_MISMATCH;
  if (x.dtype() != weight.dtype() || x.dtype() != result.dtype())
    return core::Status::UNSUPPORTED_TYPE;
  if (has_bias && bias.dtype() != x.dtype())
    return core::Status::UNSUPPORTED_TYPE;
  std::vector<size_t> out_shape = x.shape();
  out_shape.back() = out_f;
  core::Status st = result.Resize(out_shape);
  if (st != core::Status::OK) return st;
  if (rows == 0 || k == 0 || out_f == 0) return core::Status::OK;

  cudaStream_t s = core::cuda::AxonoCurrentStream();

  // ① result 初值: bias 广播 / 清零
  const size_t total = rows * out_f;
  const dim3 grid(static_cast<unsigned>((total + 255) / 256));
  if (has_bias) {
    if (x.dtype() == core::DataType::FLOAT32)
      BroadcastBiasF32<<<grid, 256, 0, s>>>(result.data<float>(),
                                            bias.data<float>(), rows, out_f);
    else
      BroadcastBiasF64<<<grid, 256, 0, s>>>(result.data<double>(),
                                            bias.data<double>(), rows, out_f);
  } else {
    if (x.dtype() == core::DataType::FLOAT32)
      FillZeroF32<<<grid, 256, 0, s>>>(result.data<float>(), total);
    else
      FillZeroF64<<<grid, 256, 0, s>>>(result.data<double>(), total);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return core::Status::DEVICE_ERROR;

  // ② gemm 累加: result += 1.0 * x @ W^T
  // 列主序映射: 行主 result(rows,out_f) 视为列主 resultᵀ(out_f,rows)。
  //   resultᵀ = Wᵀ · xᵀ: W 视为列主 (k,out_f) ld=k → opA=T;
  //   x 视为列主 (k,rows) ld=k → opB=N。
  //   gemm(opA=T, opB=N, m=out_f, n=rows, k=k), C ld=out_f。
  const int M = static_cast<int>(out_f);
  const int N = static_cast<int>(rows);
  const int K = static_cast<int>(k);
  const float alpha_f = 1.0f, beta_f = 1.0f;
  const double alpha_d = 1.0, beta_d = 1.0;
  try {
    if (x.dtype() == core::DataType::FLOAT32) {
      if (!TryLtGemmF32(M, N, K, &alpha_f, weight.data<float>(), K,
                        x.data<float>(), K, &beta_f, result.data<float>(),
                        static_cast<int>(out_f), s, CUBLAS_OP_T)) {
        cublasHandle_t handle = GetCublasHandle();
        cublasSetStream(handle, s);
        AXONO_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N,
                                       K, &alpha_f, weight.data<float>(), K,
                                       x.data<float>(), K, &beta_f,
                                       result.data<float>(),
                                       static_cast<int>(out_f)));
      }
    } else {
      if (!TryLtGemmF64(M, N, K, &alpha_d, weight.data<double>(), K,
                        x.data<double>(), K, &beta_d, result.data<double>(),
                        static_cast<int>(out_f), s, CUBLAS_OP_T)) {
        cublasHandle_t handle = GetCublasHandle();
        cublasSetStream(handle, s);
        AXONO_CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N,
                                       K, &alpha_d, weight.data<double>(), K,
                                       x.data<double>(), K, &beta_d,
                                       result.data<double>(),
                                       static_cast<int>(out_f)));
      }
    }
  } catch (const std::exception &) {
    return core::Status::INTERNAL_ERROR;
  }

  if (!core::cuda::IsCapturing() && cudaDeviceSynchronize() != cudaSuccess)
    return core::Status::INTERNAL_ERROR;
  return core::Status::OK;
}

}  // namespace cuda
}  // namespace ops
}  // namespace axono
