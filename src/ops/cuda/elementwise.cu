// 逐元素算子 (CUDA): sub/mul/div/neg/abs/exp/log/sqrt/sigmoid/tanh
// 统一模式: 模板 kernel + AxonoCurrentStream() 提交 + 捕获守卫。
// functors 须在 device 上可构造 —— 以空结构体传 kernel (零开销)。
#include <cuda_runtime.h>
#include <cstddef>

#include "axono/core/cuda/capture.h"
#include "axono/core/cuda/stream.h"
#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include "axono/ops/cuda/elementwise.h"

namespace axono {
namespace ops {
namespace cuda {

// ---------- device functors ----------
struct SubOp {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a - b;
  }
};
struct MulOp {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a * b;
  }
};
struct DivOp {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a / b;
  }
};
struct NegOp {
  template <typename T>
  __device__ T operator()(T v) const {
    return -v;
  }
};
struct AbsOp {
  template <typename T>
  __device__ T operator()(T v) const {
    return v < T(0) ? -v : v;
  }
};
struct ExpOp {
  template <typename T>
  __device__ T operator()(T v) const {
    return exp(v);
  }
};
struct LogOp {
  template <typename T>
  __device__ T operator()(T v) const {
    return log(v);
  }
};
struct SqrtOp {
  template <typename T>
  __device__ T operator()(T v) const {
    return sqrt(v);
  }
};
struct SigmoidOp {
  template <typename T>
  __device__ T operator()(T v) const {
    return T(1) / (T(1) + exp(-v));
  }
};
struct TanhOp {
  template <typename T>
  __device__ T operator()(T v) const {
    return tanh(v);
  }
};

// ---------- 全局 kernel 模板 ----------
template <typename T, typename Op>
__global__ void ElementwiseBinaryKernel(const T *a, const T *b, T *out,
                                        size_t n, Op op) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = op(a[idx], b[idx]);
}

template <typename T, typename F>
__global__ void ElementwiseUnaryKernel(const T *x, T *out, size_t n, F f) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = f(x[idx]);
}

namespace {

inline dim3 LaunchConfig(size_t n) {
  const size_t block = 256;
  return dim3((n + block - 1) / block);
}

// 捕获期间跳过同步 (kernel 已提交到捕获流)
inline core::Status SyncGuard() {
  if (!core::cuda::IsCapturing()) {
    if (cudaDeviceSynchronize() != cudaSuccess) return core::Status::DEVICE_ERROR;
  }
  return core::Status::OK;
}

template <typename Op>
inline core::Status BinaryDispatch(const core::Tensor &a, const core::Tensor &b,
                             core::Tensor &result, Op op) {
  if (!a.IsSameShape(b) || !a.IsSameShape(result))
    return core::Status::SHAPE_MISMATCH;
  if (a.dtype() != b.dtype() || a.dtype() != result.dtype())
    return core::Status::UNSUPPORTED_TYPE;
  if (!a.is_cuda()) return core::Status::DEVICE_MISMATCH;

  const size_t n = a.num_elements();
  const dim3 grid = LaunchConfig(n);
  cudaStream_t s = core::cuda::AxonoCurrentStream();
  cudaError_t err = cudaSuccess;
  switch (a.dtype()) {
    case core::DataType::FLOAT32:
      ElementwiseBinaryKernel<float><<<grid, 256, 0, s>>>(
          a.data<float>(), b.data<float>(), result.data<float>(), n, op);
      err = cudaGetLastError();
      break;
    case core::DataType::FLOAT64:
      ElementwiseBinaryKernel<double><<<grid, 256, 0, s>>>(
          a.data<double>(), b.data<double>(), result.data<double>(), n, op);
      err = cudaGetLastError();
      break;
    case core::DataType::INT32:
      ElementwiseBinaryKernel<int32_t><<<grid, 256, 0, s>>>(
          a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n, op);
      err = cudaGetLastError();
      break;
    case core::DataType::INT64:
      ElementwiseBinaryKernel<int64_t><<<grid, 256, 0, s>>>(
          a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n, op);
      err = cudaGetLastError();
      break;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }
  if (err != cudaSuccess) return core::Status::DEVICE_ERROR;
  return SyncGuard();
}

template <typename F>
inline core::Status UnaryDispatchF(const core::Tensor &x, core::Tensor &result,
                             F f) {
  if (!x.IsSameShape(result)) return core::Status::SHAPE_MISMATCH;
  if (x.dtype() != result.dtype()) return core::Status::UNSUPPORTED_TYPE;
  if (!x.is_cuda()) return core::Status::DEVICE_MISMATCH;

  const size_t n = x.num_elements();
  const dim3 grid = LaunchConfig(n);
  cudaStream_t s = core::cuda::AxonoCurrentStream();
  cudaError_t err = cudaSuccess;
  switch (x.dtype()) {
    case core::DataType::FLOAT32:
      ElementwiseUnaryKernel<float><<<grid, 256, 0, s>>>(
          x.data<float>(), result.data<float>(), n, f);
      err = cudaGetLastError();
      break;
    case core::DataType::FLOAT64:
      ElementwiseUnaryKernel<double><<<grid, 256, 0, s>>>(
          x.data<double>(), result.data<double>(), n, f);
      err = cudaGetLastError();
      break;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }
  if (err != cudaSuccess) return core::Status::DEVICE_ERROR;
  return SyncGuard();
}

core::Status UnaryOpCheck(const core::Tensor &x, core::Tensor &result) {
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;
  if (result.dtype() != x.dtype()) return core::Status::UNSUPPORTED_TYPE;
  return core::Status::OK;
}

core::Status BinaryOpCheck(const core::Tensor &a, const core::Tensor &b,
                     core::Tensor &result) {
  if (a.ndim() != b.ndim() || !a.IsSameShape(b))
    return core::Status::SHAPE_MISMATCH;
  if (a.dtype() != b.dtype()) return core::Status::UNSUPPORTED_TYPE;
  core::Status st = result.Resize(a.shape());
  if (st != core::Status::OK) return st;
  if (result.dtype() != a.dtype()) return core::Status::UNSUPPORTED_TYPE;
  return core::Status::OK;
}

}  // namespace

// ---------- 公开接口 ----------

core::Status Sub(const core::Context &ctx, const core::Tensor &a,
           const core::Tensor &b, core::Tensor &result) {
  (void)ctx;
  core::Status st = BinaryOpCheck(a, b, result);
  if (st != core::Status::OK) return st;
  return BinaryDispatch(a, b, result, SubOp{});
}

core::Status Mul(const core::Context &ctx, const core::Tensor &a,
           const core::Tensor &b, core::Tensor &result) {
  (void)ctx;
  core::Status st = BinaryOpCheck(a, b, result);
  if (st != core::Status::OK) return st;
  return BinaryDispatch(a, b, result, MulOp{});
}

core::Status Div(const core::Context &ctx, const core::Tensor &a,
           const core::Tensor &b, core::Tensor &result) {
  (void)ctx;
  core::Status st = BinaryOpCheck(a, b, result);
  if (st != core::Status::OK) return st;
  return BinaryDispatch(a, b, result, DivOp{});
}

core::Status Neg(const core::Context &ctx, const core::Tensor &x,
           core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatchF(x, result, NegOp{});
}

core::Status Abs(const core::Context &ctx, const core::Tensor &x,
           core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatchF(x, result, AbsOp{});
}

core::Status Exp(const core::Context &ctx, const core::Tensor &x,
           core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatchF(x, result, ExpOp{});
}

core::Status Log(const core::Context &ctx, const core::Tensor &x,
           core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatchF(x, result, LogOp{});
}

core::Status Sqrt(const core::Context &ctx, const core::Tensor &x,
            core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatchF(x, result, SqrtOp{});
}

core::Status Sigmoid(const core::Context &ctx, const core::Tensor &x,
               core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatchF(x, result, SigmoidOp{});
}

core::Status Tanh(const core::Context &ctx, const core::Tensor &x,
            core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatchF(x, result, TanhOp{});
}

}  // namespace cuda
}  // namespace ops
}  // namespace axono
