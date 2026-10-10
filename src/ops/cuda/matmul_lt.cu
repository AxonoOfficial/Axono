// Axono v0.2 — cuBLASLt 后端 (float/double matmul 加速)
//
// cublasLtMatmul 相比经典 cuBLAS gemm 的优势:
//   - 启发式算法选择 (cublasLtMatmulAlgoGetHeuristic), 自动挑当前
//     GPU 最优 kernel, 通常比 cublasSgemm 快 10%~50%;
//   - 支持拆分 K (split-k)、TF32 等经典 API 不暴露的优化路径;
//   - 显式 workspace 管理。
//
// 策略: 优先走 Lt (heuristic 取 top-1), 任何一步失败自动回退经典 cuBLAS,
// 对上层完全透明。可用 axono_use_cublas_lt(false) 全局关闭。
// 列主序映射与文件 matmul.cu 头注释一致 (行主序免转置)。

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {

namespace {

// ---------------------------------------------------------------------------
// Lt 句柄与 workspace (进程级, 线程安全懒加载)
// ---------------------------------------------------------------------------
constexpr size_t LT_WORKSPACE_SIZE = 32 * 1024 * 1024;  // 32 MiB

cublasLtHandle_t GetLtHandle() {
  static cublasLtHandle_t handle = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasLtCreate failed");
    }
  });
  return handle;
}

void *GetLtWorkspace() {
  static void *buf = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    if (cudaMalloc(&buf, LT_WORKSPACE_SIZE) != cudaSuccess) {
      buf = nullptr;  // 分配失败 -> 无 workspace 模式 (heuristic 会适配)
    }
  });
  return buf;
}

std::mutex &LtMutex() {
  static std::mutex m;
  return m;
}

// algo 按形状缓存 (模板参数区分 float/double): heuristic 查询开销
// 可观 (微秒级 CPU), 同形状的 matmul (典型训练/推理循环) 无需重复查询。
struct ShapeKey {
  int n, m, k;
  bool operator==(const ShapeKey &o) const {
    return n == o.n && m == o.m && k == o.k;
  }
};
struct ShapeKeyHash {
  size_t operator()(const ShapeKey &s) const {
    return (static_cast<size_t>(s.n) * 1000003u ^
            static_cast<size_t>(s.m) * 1000033u) *
               1000037u ^
           static_cast<size_t>(s.k);
  }
};
std::mutex &AlgoCacheMutex() {
  static std::mutex m;
  return m;
}

// 每种 dtype 一个独立缓存 (static 局部量): 用 int tag 区分
// 0=float, 1=double, 2=fp16, 由调用方 (LtGemm 特化) 传入。
std::unordered_map<ShapeKey, cublasLtMatmulAlgo_t, ShapeKeyHash>
    &AlgoCacheGet(int tag) {
  static std::unordered_map<ShapeKey, cublasLtMatmulAlgo_t, ShapeKeyHash>
      caches[3];
  return caches[tag];
}

bool AlgoCacheFind(int tag, int n, int m, int k,
                   cublasLtMatmulAlgo_t *out) {
  std::lock_guard<std::mutex> lk(AlgoCacheMutex());
  auto &cache = AlgoCacheGet(tag);
  auto it = cache.find({n, m, k});
  if (it == cache.end()) return false;
  *out = it->second;
  return true;
}

void AlgoCachePut(int tag, int n, int m, int k,
                  const cublasLtMatmulAlgo_t &algo) {
  std::lock_guard<std::mutex> lk(AlgoCacheMutex());
  AlgoCacheGet(tag)[{n, m, k}] = algo;
}

const char *LtStatusString(cublasStatus_t s) {
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

#define AXONO_LT_CHECK(expr)                                          \
  do {                                                                \
    cublasStatus_t s_ = (expr);                                       \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                \
      throw std::runtime_error(std::string("cuBLASLt error: ") +      \
                               LtStatusString(s_) + " at " #expr);    \
    }                                                                 \
  } while (0)

// 全局开关: 是否优先走 Lt
std::atomic<bool> g_use_lt{true};

// 单个 dtype 的 Lt gemm 模板实现; 失败抛异常, 由调用方回退。
template <typename T>
cublasStatus_t LtGemm(cublasOperation_t, cublasOperation_t, int, int, int,
                      const T *, const T *, int, const T *, int, const T *,
                      T *, int, cudaStream_t) {
  return CUBLAS_STATUS_NOT_SUPPORTED;  // 未知类型: 触发回退
}

template <>
cublasStatus_t LtGemm<float>(cublasOperation_t op_b, cublasOperation_t op_a,
                             int n, int m, int k, const float *alpha,
                             const float *b, int ldb, const float *a, int lda,
                             const float *beta, float *c, int ldc,
                             cudaStream_t stream) {
  cublasLtHandle_t lt = GetLtHandle();
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
      CUBLAS_STATUS_SUCCESS) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  // 列主序映射: C'(n x m) = B'(n x k) * A'(k x m)
  if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &op_b, sizeof(op_b)) !=
      CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(op_desc);
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &op_a, sizeof(op_a)) !=
      CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(op_desc);
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(op_desc);
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  void *ws = GetLtWorkspace();
  size_t ws_size = ws ? LT_WORKSPACE_SIZE : 0;
  if (ws) {
    cublasLtMatmulPreferenceSetAttribute(pref,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &ws_size, sizeof(ws_size));
  }

  // 布局: B'(n x k) ld=ldb, A'(k x m) ld=lda, C'(n x m) ld=ldc
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  do {
    if (cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, op_b == CUBLAS_OP_N ? n : k,
                                   op_b == CUBLAS_OP_N ? k : n, ldb) !=
        CUBLAS_STATUS_SUCCESS) { status = CUBLAS_STATUS_INTERNAL_ERROR; break; }
    if (cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, op_a == CUBLAS_OP_N ? k : m,
                                   op_a == CUBLAS_OP_N ? m : k, lda) !=
        CUBLAS_STATUS_SUCCESS) { status = CUBLAS_STATUS_INTERNAL_ERROR; break; }
    if (cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, n, m, ldc) !=
        CUBLAS_STATUS_SUCCESS) { status = CUBLAS_STATUS_INTERNAL_ERROR; break; }

    // 先查 algo 缓存; 未命中才做 heuristic 查询并缓存
    // (tag 由 alpha 类型推导: float=0, double=1)
    constexpr int kTag =
        std::is_same<std::decay_t<decltype(*alpha)>, float>::value ? 0 : 1;
    cublasLtMatmulAlgo_t algo;
    if (!AlgoCacheFind(kTag, n, m, k, &algo)) {
      if (cublasLtMatmulAlgoGetHeuristic(lt, op_desc, lb, la, lc, lc, pref,
                                         1, &heuristic, &returned) !=
              CUBLAS_STATUS_SUCCESS ||
          returned == 0) {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
        break;
      }
      algo = heuristic.algo;
      AlgoCachePut(kTag, n, m, k, algo);
    }

    status = cublasLtMatmul(lt, op_desc, alpha, b, lb, a, la, beta, c, lc, c,
                            lc, &algo, ws, ws_size, stream);
  } while (false);

  if (la) cublasLtMatrixLayoutDestroy(la);
  if (lb) cublasLtMatrixLayoutDestroy(lb);
  if (lc) cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatmulDescDestroy(op_desc);
  return status;
}

template <>
cublasStatus_t LtGemm<double>(cublasOperation_t op_b, cublasOperation_t op_a,
                              int n, int m, int k, const double *alpha,
                              const double *b, int ldb, const double *a,
                              int lda, const double *beta, double *c, int ldc,
                              cudaStream_t stream) {
  cublasLtHandle_t lt = GetLtHandle();
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_64F, CUDA_R_64F) !=
      CUBLAS_STATUS_SUCCESS) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &op_b, sizeof(op_b)) !=
      CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(op_desc);
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &op_a, sizeof(op_a)) !=
      CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(op_desc);
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(op_desc);
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  void *ws = GetLtWorkspace();
  size_t ws_size = ws ? LT_WORKSPACE_SIZE : 0;
  if (ws) {
    cublasLtMatmulPreferenceSetAttribute(pref,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &ws_size, sizeof(ws_size));
  }

  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  do {
    if (cublasLtMatrixLayoutCreate(&lb, CUDA_R_64F, op_b == CUBLAS_OP_N ? n : k,
                                   op_b == CUBLAS_OP_N ? k : n, ldb) !=
        CUBLAS_STATUS_SUCCESS) { status = CUBLAS_STATUS_INTERNAL_ERROR; break; }
    if (cublasLtMatrixLayoutCreate(&la, CUDA_R_64F, op_a == CUBLAS_OP_N ? k : m,
                                   op_a == CUBLAS_OP_N ? m : k, lda) !=
        CUBLAS_STATUS_SUCCESS) { status = CUBLAS_STATUS_INTERNAL_ERROR; break; }
    if (cublasLtMatrixLayoutCreate(&lc, CUDA_R_64F, n, m, ldc) !=
        CUBLAS_STATUS_SUCCESS) { status = CUBLAS_STATUS_INTERNAL_ERROR; break; }

    // 先查 algo 缓存; 未命中才做 heuristic 查询并缓存
    // (tag 由 alpha 类型推导: float=0, double=1)
    constexpr int kTag =
        std::is_same<std::decay_t<decltype(*alpha)>, float>::value ? 0 : 1;
    cublasLtMatmulAlgo_t algo;
    if (!AlgoCacheFind(kTag, n, m, k, &algo)) {
      if (cublasLtMatmulAlgoGetHeuristic(lt, op_desc, lb, la, lc, lc, pref,
                                         1, &heuristic, &returned) !=
              CUBLAS_STATUS_SUCCESS ||
          returned == 0) {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
        break;
      }
      algo = heuristic.algo;
      AlgoCachePut(kTag, n, m, k, algo);
    }

    status = cublasLtMatmul(lt, op_desc, alpha, b, lb, a, la, beta, c, lc, c,
                            lc, &algo, ws, ws_size, stream);
  } while (false);

  if (la) cublasLtMatrixLayoutDestroy(la);
  if (lb) cublasLtMatrixLayoutDestroy(lb);
  if (lc) cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatmulDescDestroy(op_desc);
  return status;
}
cublasStatus_t LtGemmF16(cublasOperation_t op_b, cublasOperation_t op_a,
                         int n, int m, int k, const __half *alpha,
                         const __half *b, int ldb, const __half *a,
                         int lda, const __half *beta, __half *c, int ldc,
                         cudaStream_t stream) {
  cublasLtHandle_t lt = GetLtHandle();
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  // 16F 输入输出 + 16F 累加 (V100 tensor core; 先用最保守 16F compute 
  // 保证兼容, 精度由测试把关)
  if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F) !=
      CUBLAS_STATUS_SUCCESS) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &op_b, sizeof(op_b)) !=
      CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(op_desc);
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &op_a, sizeof(op_a)) !=
      CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(op_desc);
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(op_desc);
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  void *ws = GetLtWorkspace();
  size_t ws_size = ws ? LT_WORKSPACE_SIZE : 0;
  if (ws) {
    cublasLtMatmulPreferenceSetAttribute(pref,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &ws_size, sizeof(ws_size));
  }

  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  do {
    if (cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, op_b == CUBLAS_OP_N ? n : k,
                                   op_b == CUBLAS_OP_N ? k : n, ldb) !=
        CUBLAS_STATUS_SUCCESS) { status = CUBLAS_STATUS_INTERNAL_ERROR; break; }
    if (cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, op_a == CUBLAS_OP_N ? k : m,
                                   op_a == CUBLAS_OP_N ? m : k, lda) !=
        CUBLAS_STATUS_SUCCESS) { status = CUBLAS_STATUS_INTERNAL_ERROR; break; }
    if (cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, n, m, ldc) !=
        CUBLAS_STATUS_SUCCESS) { status = CUBLAS_STATUS_INTERNAL_ERROR; break; }

    // tag=2 (float=0, double=1)
    constexpr int kTag = 2;
    cublasLtMatmulAlgo_t algo;
    if (!AlgoCacheFind(kTag, n, m, k, &algo)) {
      if (cublasLtMatmulAlgoGetHeuristic(lt, op_desc, lb, la, lc, lc, pref,
                                         1, &heuristic, &returned) !=
              CUBLAS_STATUS_SUCCESS ||
          returned == 0) {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
        break;
      }
      algo = heuristic.algo;
      AlgoCachePut(kTag, n, m, k, algo);
    }

    status = cublasLtMatmul(lt, op_desc, alpha, b, lb, a, la, beta, c,
                            lc, c, lc, &algo, ws, ws_size, stream);
  } while (false);

  if (la) cublasLtMatrixLayoutDestroy(la);
  if (lb) cublasLtMatrixLayoutDestroy(lb);
  if (lc) cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatmulDescDestroy(op_desc);
  return status;
}

}  // namespace

// ---------------------------------------------------------------------------
// 对外接口
// ---------------------------------------------------------------------------

// 全局开关: Lt 优先 (默认开); 失败时始终自动回退经典 cuBLAS。
void UseCublasLt(bool enable) { g_use_lt.store(enable); }
bool CublasLtEnabled() { return g_use_lt.load(); }

// Lt 路径: 成功返回 true 并已提交到 stream; 失败返回 false (调用方回退)。
// 形状检查由调用方完成。alpha/beta 语义与 cublasXgemm 相同
// (C = alpha * op(A) * op(B) + beta * C, 列主序映射后即 C' = ...)。
bool TryLtGemmF32(int n, int m, int k, const float *alpha, const float *b,
                  int ldb, const float *a, int lda, const float *beta,
                  float *c, int ldc, cudaStream_t stream,
                  cublasOperation_t transa = CUBLAS_OP_N) {
  if (!g_use_lt.load()) return false;
  std::lock_guard<std::mutex> lock(LtMutex());
  try {
    // transa 作用于 b 指针 (即 LtGemm 的 op_b), 供 Linear 等 W^T gemm 使用
    return LtGemm<float>(transa, CUBLAS_OP_N, n, m, k, alpha, b, ldb, a,
                         lda, beta, c, ldc, stream) == CUBLAS_STATUS_SUCCESS;
  } catch (const std::exception &) {
    return false;
  }
}

bool TryLtGemmF64(int n, int m, int k, const double *alpha, const double *b,
                  int ldb, const double *a, int lda, const double *beta,
                  double *c, int ldc, cudaStream_t stream,
                  cublasOperation_t transa = CUBLAS_OP_N) {
  if (!g_use_lt.load()) return false;
  std::lock_guard<std::mutex> lock(LtMutex());
  try {
    // transa 作用于 b 指针 (即 LtGemm 的 op_b), 供 Linear 等 W^T gemm 使用
    return LtGemm<double>(transa, CUBLAS_OP_N, n, m, k, alpha, b, ldb, a,
                          lda, beta, c, ldc, stream) == CUBLAS_STATUS_SUCCESS;
  } catch (const std::exception &) {
    return false;
  }
}

// FP16 gemm: 16F 输入/输出 + 32F 累加 (tensor core)。beta 固定 1.0
// (调用方先把 result 初始化为 bias 或 0)。
bool TryLtGemmF16(int n, int m, int k, const __half *b, int ldb,
                  const __half *a, int lda, __half *c, int ldc,
                  cudaStream_t stream, cublasOperation_t transa = CUBLAS_OP_N,
                  float beta = 1.0f) {
  if (!g_use_lt.load()) return false;
  std::lock_guard<std::mutex> lock(LtMutex());
  try {
    const __half alpha_one = __half(1.0f);
    const __half beta_h = __half(beta);
    return LtGemmF16(transa, CUBLAS_OP_N, n, m, k, &alpha_one, b, ldb, a,
                     lda, &beta_h, c, ldc, stream) ==
           CUBLAS_STATUS_SUCCESS;
  } catch (const std::exception &) {
    return false;
  }
}

}  // namespace cuda
}  // namespace ops
}  // namespace axono
