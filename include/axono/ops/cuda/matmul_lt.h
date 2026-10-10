// Axono v0.2 — cuBLASLt 接口声明 (实现在 matmul_lt.cu)
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "axono/core/macros.h"

namespace axono {
namespace ops {
namespace cuda {

// 全局开关: matmul 是否优先走 cuBLASLt (默认开)。
// Lt 不可用/失败时始终自动回退经典 cuBLAS, 对上层透明。
AXONO_EXPORT void UseCublasLt(bool enable);
AXONO_EXPORT bool CublasLtEnabled();

// Lt 单次 gemm (float/double); 返回是否成功提交到 stream。
// 参数语义与 cublasSgemm/Dgemm 的列主序映射一致 (见 matmul.cu 头注释):
//   C'(n x m) = B'(n x k) * A'(k x m)
AXONO_EXPORT bool TryLtGemmF32(int n, int m, int k, const float *alpha,
                               const float *b, int ldb, const float *a,
                               int lda, const float *beta, float *c, int ldc,
                               cudaStream_t stream,
                               cublasOperation_t transa = CUBLAS_OP_N);
AXONO_EXPORT bool TryLtGemmF64(int n, int m, int k, const double *alpha,
                               const double *b, int ldb, const double *a,
                               int lda, const double *beta, double *c, int ldc,
                               cudaStream_t stream,
                               cublasOperation_t transa = CUBLAS_OP_N);

}  // namespace cuda
}  // namespace ops
}  // namespace axono
