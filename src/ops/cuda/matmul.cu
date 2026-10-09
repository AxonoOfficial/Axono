// Axono v0.2 — CUDA 矩阵乘法 (cuBLAS 后端)
// 用 cuBLAS gemm 取代 dev 分支的手写 naive/tiled kernel:
//   - float32  -> cublasSgemm
//   - float64  -> cublasDgemm
// 其余类型回退到手写 tiled kernel (int32)。
// cuBLAS 是列主序: 行主序 C(m x n) = A(m x k) * B(k x n) 等价于列主序
// C'(n x m) = B'(n x k) * A'(k x m)，其中 X' 即 X 的行主序内存重新解释，
// 因此无需真实转置或任何数据拷贝。

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

namespace {

// -----------------------------------------------------------------------
// cuBLAS 句柄管理 (线程安全懒加载)
// -----------------------------------------------------------------------
cublasHandle_t GetCublasHandle() {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasCreate failed");
    }
  }
  return handle;
}

const char *CublasStatusString(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    default: return "UNKNOWN";
  }
}

#define AXONO_CUBLAS_CHECK(expr)                                        \
  do {                                                                  \
    cublasStatus_t s_ = (expr);                                         \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                  \
      throw std::runtime_error(std::string("cuBLAS error: ") +          \
                               CublasStatusString(s_) + " at " #expr);  \
    }                                                                   \
  } while (0)

// -----------------------------------------------------------------------
// int32 回退 kernel (cuBLAS 不支持整型 gemm)
// -----------------------------------------------------------------------
template <typename T, size_t BLOCK_SIZE = 16>
__global__ void MatMulTiledKernel(const T *__restrict__ a,
                                  const T *__restrict__ b,
                                  T *__restrict__ result, size_t m, size_t n,
                                  size_t k) {
  __shared__ T a_tile[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T b_tile[BLOCK_SIZE][BLOCK_SIZE];

  size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  T sum = 0;
  for (size_t tile = 0; tile < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
    size_t ac = tile * BLOCK_SIZE + threadIdx.x;
    size_t br = tile * BLOCK_SIZE + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && ac < k) ? a[row * k + ac] : T(0);
    b_tile[threadIdx.y][threadIdx.x] =
        (br < k && col < n) ? b[br * n + col] : T(0);
    __syncthreads();
    for (size_t l = 0; l < BLOCK_SIZE; ++l) {
      sum += a_tile[threadIdx.y][l] * b_tile[l][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < m && col < n) result[row * n + col] = sum;
}

core::Status Int32MatMul(const core::Tensor &a, const core::Tensor &b,
                         core::Tensor &result) {
  size_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
  constexpr size_t BLOCK = 16;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((n + BLOCK - 1) / BLOCK, (m + BLOCK - 1) / BLOCK);
  MatMulTiledKernel<int32_t><<<grid, block>>>(a.data<int32_t>(),
                                              b.data<int32_t>(),
                                              result.data<int32_t>(), m, n, k);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return core::Status::INTERNAL_ERROR;
  }
  return cudaDeviceSynchronize() == cudaSuccess ? core::Status::OK
                                                : core::Status::INTERNAL_ERROR;
}

}  // namespace

// -----------------------------------------------------------------------
// 对外接口
// -----------------------------------------------------------------------
core::Status MatMul(const core::Context &ctx, const core::Tensor &a,
                    const core::Tensor &b, core::Tensor &result) {
  (void)ctx;

  if (a.ndim() != 2 || b.ndim() != 2) return core::Status::INVALID_ARGUMENT;
  if (a.shape()[1] != b.shape()[0]) return core::Status::SHAPE_MISMATCH;
  if (a.dtype() != b.dtype()) return core::Status::UNSUPPORTED_TYPE;
  if (result.dtype() != a.dtype()) return core::Status::UNSUPPORTED_TYPE;

  const size_t m = a.shape()[0];
  const size_t k = a.shape()[1];
  const size_t n = b.shape()[1];

  // 列主序映射 (见文件头注释):
  //   A 行主序 (m x k) -> 视为列主序 (k x m), 主维 lda = k
  //   B 行主序 (k x n) -> 视为列主序 (n x k), 主维 ldb = n
  //   C 行主序 (m x n) -> 视为列主序 (n x m), 主维 ldc = n
  const size_t lda = k;
  const size_t ldb = n;
  const size_t ldc = n;

  try {
    cublasHandle_t handle = GetCublasHandle();
    switch (a.dtype()) {
      case core::DataType::FLOAT32: {
        const float alpha = 1.0f, beta = 0.0f;
        AXONO_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m,
                                       k, &alpha, b.data<float>(), ldb,
                                       a.data<float>(), lda, &beta,
                                       result.data<float>(), ldc));
        break;
      }
      case core::DataType::FLOAT64: {
        const double alpha = 1.0, beta = 0.0;
        AXONO_CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m,
                                       k, &alpha, b.data<double>(), ldb,
                                       a.data<double>(), lda, &beta,
                                       result.data<double>(), ldc));
        break;
      }
      case core::DataType::INT32:
        return Int32MatMul(a, b, result);
      default:
        return core::Status::UNSUPPORTED_TYPE;
    }
  } catch (const std::exception &) {
    return core::Status::INTERNAL_ERROR;
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
    return core::Status::INTERNAL_ERROR;
  }
  return core::Status::OK;
}

core::Status MatMulAccumulate(const core::Context &ctx, const core::Tensor &a,
                              const core::Tensor &b, core::Tensor &result) {
  (void)ctx;

  if (a.ndim() != 2 || b.ndim() != 2) return core::Status::INVALID_ARGUMENT;
  if (a.shape()[1] != b.shape()[0]) return core::Status::SHAPE_MISMATCH;
  if (a.dtype() != b.dtype() || result.dtype() != a.dtype())
    return core::Status::UNSUPPORTED_TYPE;
  if (result.shape()[0] != a.shape()[0] || result.shape()[1] != b.shape()[1])
    return core::Status::SHAPE_MISMATCH;

  const size_t m = a.shape()[0];
  const size_t k = a.shape()[1];
  const size_t n = b.shape()[1];
  const size_t lda = k, ldb = n, ldc = n;

  // 累加语义 (beta=1) 要求 result != a 且 result != b: 否则会边读边写。
  // 调用方 (绑定层) 需保证这一点。
  try {
    cublasHandle_t handle = GetCublasHandle();
    switch (a.dtype()) {
      case core::DataType::FLOAT32: {
        const float alpha = 1.0f, beta = 1.0f;
        AXONO_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m,
                                       k, &alpha, b.data<float>(), ldb,
                                       a.data<float>(), lda, &beta,
                                       result.data<float>(), ldc));
        break;
      }
      case core::DataType::FLOAT64: {
        const double alpha = 1.0, beta = 1.0;
        AXONO_CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m,
                                       k, &alpha, b.data<double>(), ldb,
                                       a.data<double>(), lda, &beta,
                                       result.data<double>(), ldc));
        break;
      }
      default:
        // 整型无 cuBLAS: 先算到临时再累加 (仍是一次额外分配, 非常规路径)
        return core::Status::UNSUPPORTED_TYPE;
    }
  } catch (const std::exception &) {
    return core::Status::INTERNAL_ERROR;
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
    return core::Status::INTERNAL_ERROR;
  }
  return core::Status::OK;
}

}  // namespace cuda
}  // namespace ops
}  // namespace axono
