#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>

#include "axono/ops/cpu/add.h"

#include "axono/core/macros.h"
#include "axono/core/tensor.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || \
    defined(_M_X64)
#define AXONO_USE_X86_INTRINSICS 1
#include <immintrin.h>
#else
#define AXONO_USE_X86_INTRINSICS 0
#endif

namespace axono {
namespace ops {
namespace cpu {

constexpr size_t BLOCK_SIZE_M = 512;
constexpr size_t BLOCK_SIZE_N = 512;
constexpr size_t BLOCK_SIZE_K = 256;

// 寄存器分块大小
constexpr size_t REGISTER_BLOCK = 4;

constexpr size_t MR = 8;    // 微内核高
constexpr size_t NR = 8;    // 微内核宽
constexpr size_t KC = 256;  // 沿 K 方向的 pack 长度

alignas(64) static float packA_buf[MR * KC];  // 静态缓冲区
alignas(64) static float packB_buf[NR * KC];

inline void pack_a_block(const float *a, size_t lda, size_t i0, size_t p0,
                         size_t mr, size_t kc) {
  const float *__restrict src = a + i0 * lda + p0;
  float *__restrict dst = packA_buf;
  for (size_t j = 0; j < kc; ++j) {
    for (size_t i = 0; i < mr; ++i) {
      dst[i] = src[i * lda];
    }
    dst += mr;
    src += 1;
  }
}

// 把 B 的子块 [KC x NR] 按行 pack 到 packB_buf（无需转置）
inline void pack_b_block(const float *b, size_t ldb, size_t p0, size_t j0,
                         size_t kc, size_t nr) {
  const float *__restrict src = b + p0 * ldb + j0;
  float *__restrict dst = packB_buf;
  for (size_t p = 0; p < kc; ++p) {
    std::memcpy(dst, src, nr * sizeof(float));
    dst += nr;
    src += ldb;
  }
}

// 只对float类型使用AVX优化的sgemm内核
#if AXONO_USE_X86_INTRINSICS
extern "C" void sgemm_kernel_16x6(const float *A, const float *B, float *C,
                                  std::size_t k, std::size_t lda,
                                  std::size_t ldb, std::size_t ldc, float alpha,
                                  float beta) {
  __m256 va;
  __m256 vb0, vb1, vb2, vb3, vb4, vb5;
  __m256 c0, c1, c2, c3, c4, c5, c6, c7;
  __m256 c8, c9, c10, c11, c12, c13, c14, c15;

  /* 16 行结果寄存器清零 */
  c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = c11 = c12 = c13 =
      c14 = c15 = _mm256_setzero_ps();

  for (std::size_t p = 0; p < k; ++p) {
    /* 加载 B 的 6 个元素并广播 */
    vb0 = _mm256_set1_ps(B[0]);
    vb1 = _mm256_set1_ps(B[1]);
    vb2 = _mm256_set1_ps(B[2]);
    vb3 = _mm256_set1_ps(B[3]);
    vb4 = _mm256_set1_ps(B[4]);
    vb5 = _mm256_set1_ps(B[5]);
    B += ldb;  // B 下一行

/* 16 行 × 6 列累加，完全展开 */
#define COMPUTE_ROW(row)                     \
  va = _mm256_loadu_ps(A + row * lda);       \
  c##row = _mm256_fmadd_ps(va, vb0, c##row); \
  c##row = _mm256_fmadd_ps(va, vb1, c##row); \
  c##row = _mm256_fmadd_ps(va, vb2, c##row); \
  c##row = _mm256_fmadd_ps(va, vb3, c##row); \
  c##row = _mm256_fmadd_ps(va, vb4, c##row); \
  c##row = _mm256_fmadd_ps(va, vb5, c##row);

    // clang-format off
    COMPUTE_ROW(0)
    COMPUTE_ROW(1)
    COMPUTE_ROW(2)
    COMPUTE_ROW(3)
    COMPUTE_ROW(4)
    COMPUTE_ROW(5)
    COMPUTE_ROW(6)
    COMPUTE_ROW(7)
    COMPUTE_ROW(8)
    COMPUTE_ROW(9) COMPUTE_ROW(10) COMPUTE_ROW(11) COMPUTE_ROW(12)
        COMPUTE_ROW(13) COMPUTE_ROW(14) COMPUTE_ROW(15)
  //clang-format on
#undef COMPUTE_ROW

            A += 1;  // A 下一列
  }

  /* 把 16×6 结果写回 C，带 alpha/beta scaling */
  auto store_col = [&](__m256 v, float *addr) {
    __m256 c_old = _mm256_loadu_ps(addr);
    __m256 c_new = _mm256_fmadd_ps(v, _mm256_set1_ps(alpha),
                                   _mm256_mul_ps(c_old, _mm256_set1_ps(beta)));
    _mm256_storeu_ps(addr, c_new);
  };

  for (int i = 0; i < 16; ++i) {
    float *ci = C + i * ldc;
    __m256 res = i == 0    ? c0
                 : i == 1  ? c1
                 : i == 2  ? c2
                 : i == 3  ? c3
                 : i == 4  ? c4
                 : i == 5  ? c5
                 : i == 6  ? c6
                 : i == 7  ? c7
                 : i == 8  ? c8
                 : i == 9  ? c9
                 : i == 10 ? c10
                 : i == 11 ? c11
                 : i == 12 ? c12
                 : i == 13 ? c13
                 : i == 14 ? c14
                           : c15;

    /* 6 列依次写回 */
    store_col(res, ci + 0 * 8);
    store_col(res, ci + 1 * 8);
    store_col(res, ci + 2 * 8);
    store_col(res, ci + 3 * 8);
    store_col(res, ci + 4 * 8);
    store_col(res, ci + 5 * 8);
  }
}
#endif

// 基础优化版本 - 修复内存访问
template <typename T>
void MatMulBasicOptimized(const T *a, const T *b, T *result, size_t m, size_t n,
                          size_t k) {
  // 初始化结果为0
  std::fill(result, result + m * n, T(0));

  const size_t lda = k;  // A的行主序步长
  const size_t ldb = n;  // B的行主序步长

  for (size_t i = 0; i < m; i += BLOCK_SIZE_M) {
    size_t i_end = std::min(i + BLOCK_SIZE_M, m);
    for (size_t p = 0; p < k; p += BLOCK_SIZE_K) {
      size_t p_end = std::min(p + BLOCK_SIZE_K, k);
      for (size_t j = 0; j < n; j += BLOCK_SIZE_N) {
        size_t j_end = std::min(j + BLOCK_SIZE_N, n);

        const size_t kc_block = std::min(KC, p_end - p);
        for (size_t pp = p; pp < p_end; pp += kc_block) {
          size_t pc_cnt = std::min(kc_block, p_end - pp);

          /* 1. 先 pack B 的 [pc_cnt x NR] 子块 */
          size_t jj = j;
          for (; jj + NR <= j_end; jj += NR) {
            if constexpr (std::is_same_v<T, float>) {
              pack_b_block(reinterpret_cast<const float *>(b), ldb, pp, jj,
                           pc_cnt, NR);
            }

            /* 2. 再 pack A 的 [MR x pc_cnt] 子块，并立即计算 */
            size_t ii = i;
            for (; ii + MR <= i_end; ii += MR) {
              if constexpr (std::is_same_v<T, float>) {
                pack_a_block(reinterpret_cast<const float *>(a), lda, ii, pp,
                             MR, pc_cnt);
              }

              /* 3. 调 micro-kernel */
              if constexpr (std::is_same_v<T, float>) {
                // 使用 pack 缓冲区的优化版本
                for (size_t mi = 0; mi < MR; ++mi) {
                  const float *__restrict pa = packA_buf + mi;
                  for (size_t ni = 0; ni < NR; ++ni) {
                    const float *__restrict pb = packB_buf + ni;
                    float sum = 0.0f;
                    for (size_t kki = 0; kki < pc_cnt; ++kki) {
                      sum += pa[kki * MR] * pb[kki * NR];
                    }
                    reinterpret_cast<float *>(
                        result)[(ii + mi) * n + jj + ni] += sum;
                  }
                }
              } else {
                // 通用版本
                for (size_t mi = 0; mi < MR; ++mi) {
                  for (size_t ni = 0; ni < NR; ++ni) {
                    T sum = T(0);
                    for (size_t kki = 0; kki < pc_cnt; ++kki) {
                      sum += a[(ii + mi) * lda + pp + kki] *
                             b[(pp + kki) * ldb + jj + ni];
                    }
                    result[(ii + mi) * n + jj + ni] += sum;
                  }
                }
              }
            }

            /* 4. 边缘 mr < MR 用标量回退 */
            for (; ii < i_end; ++ii) {
              for (size_t ni = 0; ni < NR; ++ni) {
                T sum = T(0);
                for (size_t kki = 0; kki < pc_cnt; ++kki) {
                  sum += a[ii * lda + pp + kki] * b[(pp + kki) * ldb + jj + ni];
                }
                result[ii * n + jj + ni] += sum;
              }
            }
          }

          /* 5. 边缘 nr < NR 用标量回退 */
          for (; jj < j_end; ++jj) {
            for (size_t ii = i; ii < i_end; ++ii) {
              T sum = T(0);
              for (size_t kki = 0; kki < pc_cnt; ++kki) {
                sum += a[ii * lda + pp + kki] * b[(pp + kki) * ldb + jj];
              }
              result[ii * n + jj] += sum;
            }
          }
        }
      }
    }
  }
}

// SIMD优化的float版本
#if AXONO_USE_X86_INTRINSICS
template <>
void MatMulBasicOptimized<float>(const float *a, const float *b, float *result,
                                 size_t m, size_t n, size_t k) {
  constexpr size_t SIMD_WIDTH = 8;  // AVX2: 8个float

  // 初始化结果为0
  std::fill(result, result + m * n, 0.0f);

  for (size_t i = 0; i < m; i += BLOCK_SIZE_M) {
    size_t i_end = std::min(i + BLOCK_SIZE_M, m);
    for (size_t p = 0; p < k; p += BLOCK_SIZE_K) {
      size_t p_end = std::min(p + BLOCK_SIZE_K, k);
      for (size_t j = 0; j < n; j += BLOCK_SIZE_N) {
        size_t j_end = std::min(j + BLOCK_SIZE_N, n);

        // 寄存器分块 + SIMD
        for (size_t ii = i; ii < i_end; ii += REGISTER_BLOCK) {
          size_t ii_end = std::min(ii + REGISTER_BLOCK, i_end);

          for (size_t pp = p; pp < p_end; ++pp) {
            for (size_t ii2 = ii; ii2 < ii_end; ++ii2) {
              float a_val = a[ii2 * k + pp];
              __m256 a_vec = _mm256_set1_ps(a_val);

              size_t jj = j;
              // SIMD处理主要部分
              for (; jj + SIMD_WIDTH <= j_end; jj += SIMD_WIDTH) {
                __m256 b_vec = _mm256_loadu_ps(&b[pp * n + jj]);
                __m256 r_vec = _mm256_loadu_ps(&result[ii2 * n + jj]);
                r_vec = _mm256_fmadd_ps(a_vec, b_vec, r_vec);
                _mm256_storeu_ps(&result[ii2 * n + jj], r_vec);
              }

              // 处理剩余部分
              for (; jj < j_end; ++jj) {
                result[ii2 * n + jj] += a_val * b[pp * n + jj];
              }
            }
          }
        }
      }
    }
  }
}
#endif

// 修复的多线程版本
template <typename T>
void MatMulParallel(const T *a, const T *b, T *result, size_t m, size_t n,
                    size_t k) {
  constexpr size_t BLOCK_SIZE = 64;

  // 先初始化整个结果为0
  std::fill(result, result + m * n, T(0));

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (size_t i = 0; i < m; i += BLOCK_SIZE) {
    for (size_t j = 0; j < n; j += BLOCK_SIZE) {
      size_t i_end = std::min(i + BLOCK_SIZE, m);
      size_t j_end = std::min(j + BLOCK_SIZE, n);

      // 每个线程处理自己的块，没有数据竞争
      for (size_t p = 0; p < k; p += BLOCK_SIZE) {
        size_t p_end = std::min(p + BLOCK_SIZE, k);

        for (size_t ii = i; ii < i_end; ++ii) {
          for (size_t pp = p; pp < p_end; ++pp) {
            T a_val = a[ii * k + pp];
            for (size_t jj = j; jj < j_end; ++jj) {
              // 每个线程只写自己负责的块，安全
              result[ii * n + jj] += a_val * b[pp * n + jj];
            }
          }
        }
      }
    }
  }
}

// 优化的多线程版本 - 更好的负载均衡
template <typename T>
void MatMulParallelOptimized(const T *a, const T *b, T *result, size_t m,
                             size_t n, size_t k) {
  // 初始化结果为0
  std::fill(result, result + m * n, T(0));

#pragma omp parallel
  {
// 每个线程处理不同的i块
#pragma omp for schedule(static)
    for (size_t i = 0; i < m; i += BLOCK_SIZE_M) {
      size_t i_end = std::min(i + BLOCK_SIZE_M, m);

      for (size_t p = 0; p < k; p += BLOCK_SIZE_K) {
        size_t p_end = std::min(p + BLOCK_SIZE_K, k);
        for (size_t j = 0; j < n; j += BLOCK_SIZE_N) {
          size_t j_end = std::min(j + BLOCK_SIZE_N, n);

          for (size_t ii = i; ii < i_end; ++ii) {
            for (size_t pp = p; pp < p_end; ++pp) {
              T a_val = a[ii * k + pp];
              for (size_t jj = j; jj < j_end; ++jj) {
                result[ii * n + jj] += a_val * b[pp * n + jj];
              }
            }
          }
        }
      }
    }
  }
}

// 主内核函数 - 根据情况选择最优实现
template <typename T>
void MatMulOptimizedKernel(const T *a, const T *b, T *result, size_t m,
                           size_t n, size_t k) {
  // 根据矩阵大小选择不同策略
  if (m * n * k > 2000000) {  // 大矩阵用多线程
    MatMulParallelOptimized(a, b, result, m, n, k);
  } else {  // 中小矩阵用 SIMD 优化的单线程版本
    MatMulBasicOptimized(a, b, result, m, n, k);
  }
}

core::Status MatMul(const core::Context &ctx, const core::Tensor &a,
                    const core::Tensor &b, core::Tensor &result) {
  (void)ctx;  // 暂时未使用

  // 基本参数检查
  if (a.ndim() != 2 || b.ndim() != 2) {
    return core::Status::INVALID_ARGUMENT;
  }

  auto a_shape = a.shape();
  auto b_shape = b.shape();

  // 检查矩阵乘法形状兼容性
  if (a_shape[1] != b_shape[0]) {
    return core::Status::SHAPE_MISMATCH;
  }

  // 检查数据类型一致性
  if (a.dtype() != b.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 设置结果张量的形状
  std::vector<size_t> result_shape = {a_shape[0], b_shape[1]};
  core::Status status = result.Resize(result_shape);
  if (status != core::Status::OK) {
    return status;
  }

  // 设置结果的数据类型
  if (result.dtype() != a.dtype()) {
    result = core::Tensor(a.dtype(), result_shape);
    if (!result.data()) return core::Status::OUT_OF_MEMORY;
  }

  // 获取数据指针
  const void *a_data = a.data();
  const void *b_data = b.data();
  void *result_data = result.data();

  // 根据数据类型调用优化的矩阵乘法内核
  switch (a.dtype()) {
    case core::DataType::FLOAT32:
      MatMulOptimizedKernel<float>(static_cast<const float *>(a_data),
                                   static_cast<const float *>(b_data),
                                   static_cast<float *>(result_data),
                                   a_shape[0], b_shape[1], a_shape[1]);
      break;
    case core::DataType::FLOAT64:
      MatMulOptimizedKernel<double>(static_cast<const double *>(a_data),
                                    static_cast<const double *>(b_data),
                                    static_cast<double *>(result_data),
                                    a_shape[0], b_shape[1], a_shape[1]);
      break;
    case core::DataType::INT32:
      MatMulOptimizedKernel<int32_t>(static_cast<const int32_t *>(a_data),
                                     static_cast<const int32_t *>(b_data),
                                     static_cast<int32_t *>(result_data),
                                     a_shape[0], b_shape[1], a_shape[1]);
      break;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }

  return core::Status::OK;
}

// 累加变体: result += a @ b。
// CPU 上无法完全免临时: 乘积需先算出再 elementwise 加回 result,
// 否则 result 会同时作为读源与写目标被覆盖。
core::Status MatMulAccumulate(const core::Context &ctx, const core::Tensor &a,
                              const core::Tensor &b, core::Tensor &result) {
  (void)ctx;

  if (a.ndim() != 2 || b.ndim() != 2) {
    return core::Status::INVALID_ARGUMENT;
  }
  if (a.shape()[1] != b.shape()[0]) {
    return core::Status::SHAPE_MISMATCH;
  }
  if (a.dtype() != b.dtype() || result.dtype() != a.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }
  if (result.shape()[0] != a.shape()[0] || result.shape()[1] != b.shape()[1]) {
    return core::Status::SHAPE_MISMATCH;
  }

  core::Tensor prod(a.dtype(),
                    std::vector<size_t>{a.shape()[0], b.shape()[1]},
                    a.device());
  core::Status st = MatMul(ctx, a, b, prod);
  if (st != core::Status::OK) return st;
  return AddInto(ctx, result, prod, result);
}

}  // namespace cpu
}  // namespace ops
}  // namespace axono
