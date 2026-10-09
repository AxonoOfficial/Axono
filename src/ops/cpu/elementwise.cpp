// 逐元素算子 (CPU): sub/mul/div/neg/abs/exp/log/sqrt/sigmoid/tanh
// 统一模式: 模板 kernel (小规模单线程, ≥16384 OpenMP) + dtype 分派。
#include <cmath>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include "axono/ops/cpu/elementwise.h"

namespace axono {
namespace ops {
namespace cpu {

namespace {

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

core::Status UnaryOpCheck(const core::Tensor &x, core::Tensor &result) {
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;
  if (result.dtype() != x.dtype()) return core::Status::UNSUPPORTED_TYPE;
  return core::Status::OK;
}

// ---------- 二元 kernel ----------
template <typename T, typename Op>
inline void BinaryKernel(const T *a, const T *b, T *out, size_t n, Op op) {
  if (n < 16384) {
    for (size_t i = 0; i < n; ++i) out[i] = op(a[i], b[i]);
    return;
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) out[i] = op(a[i], b[i]);
}

struct OpSub {
  template <typename T>
  T operator()(T a, T b) const {
    return a - b;
  }
};
struct OpMul {
  template <typename T>
  T operator()(T a, T b) const {
    return a * b;
  }
};
struct OpDiv {
  template <typename T>
  T operator()(T a, T b) const {
    return a / b;
  }
};

// ---------- 一元 kernel (仅浮点) ----------
template <typename T, typename F>
inline void UnaryKernel(const T *x, T *out, size_t n, F &&f) {
  if (n < 16384) {
    for (size_t i = 0; i < n; ++i) out[i] = f(x[i]);
    return;
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) out[i] = f(x[i]);
}

// dtype 分派 (二元: 4 种整型/浮点; 一元: 仅浮点)
template <typename F>
inline core::Status BinaryDispatch(const core::Tensor &a, const core::Tensor &b,
                             core::Tensor &result, F &&fn) {
  if (a.dtype() != b.dtype() || a.dtype() != result.dtype())
    return core::Status::UNSUPPORTED_TYPE;
  const size_t n = a.num_elements();
  switch (a.dtype()) {
    case core::DataType::FLOAT32:
      fn(a.data<float>(), b.data<float>(), result.data<float>(), n);
      return core::Status::OK;
    case core::DataType::FLOAT64:
      fn(a.data<double>(), b.data<double>(), result.data<double>(), n);
      return core::Status::OK;
    case core::DataType::INT32:
      fn(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
      return core::Status::OK;
    case core::DataType::INT64:
      fn(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
      return core::Status::OK;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }
}

template <typename F>
inline core::Status UnaryDispatch(const core::Tensor &x, core::Tensor &result,
                            F &&fn) {
  if (x.dtype() != result.dtype()) return core::Status::UNSUPPORTED_TYPE;
  const size_t n = x.num_elements();
  switch (x.dtype()) {
    case core::DataType::FLOAT32:
      fn(x.data<float>(), result.data<float>(), n);
      return core::Status::OK;
    case core::DataType::FLOAT64:
      fn(x.data<double>(), result.data<double>(), n);
      return core::Status::OK;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }
}

}  // namespace

// ---------- 公开接口 ----------

core::Status Sub(const core::Context &ctx, const core::Tensor &a,
           const core::Tensor &b, core::Tensor &result) {
  (void)ctx;
  core::Status st = BinaryOpCheck(a, b, result);
  if (st != core::Status::OK) return st;
  return BinaryDispatch(a, b, result,
                        [](const auto *pa, const auto *pb, auto *po, size_t n) {
                          BinaryKernel(pa, pb, po, n, OpSub{});
                        });
}

core::Status Mul(const core::Context &ctx, const core::Tensor &a,
           const core::Tensor &b, core::Tensor &result) {
  (void)ctx;
  core::Status st = BinaryOpCheck(a, b, result);
  if (st != core::Status::OK) return st;
  return BinaryDispatch(a, b, result,
                        [](const auto *pa, const auto *pb, auto *po, size_t n) {
                          BinaryKernel(pa, pb, po, n, OpMul{});
                        });
}

core::Status Div(const core::Context &ctx, const core::Tensor &a,
           const core::Tensor &b, core::Tensor &result) {
  (void)ctx;
  core::Status st = BinaryOpCheck(a, b, result);
  if (st != core::Status::OK) return st;
  return BinaryDispatch(a, b, result,
                        [](const auto *pa, const auto *pb, auto *po, size_t n) {
                          BinaryKernel(pa, pb, po, n, OpDiv{});
                        });
}

core::Status Neg(const core::Context &ctx, const core::Tensor &x,
           core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatch(x, result, [](const auto *px, auto *po, size_t n) {
    UnaryKernel(px, po, n, [](auto v) { return -v; });
  });
}

core::Status Abs(const core::Context &ctx, const core::Tensor &x,
           core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatch(x, result, [](const auto *px, auto *po, size_t n) {
    UnaryKernel(px, po, n, [](auto v) { return v < 0 ? -v : v; });
  });
}

core::Status Exp(const core::Context &ctx, const core::Tensor &x,
           core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatch(x, result, [](const auto *px, auto *po, size_t n) {
    UnaryKernel(px, po, n, [](auto v) { return std::exp(v); });
  });
}

core::Status Log(const core::Context &ctx, const core::Tensor &x,
           core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatch(x, result, [](const auto *px, auto *po, size_t n) {
    UnaryKernel(px, po, n, [](auto v) { return std::log(v); });
  });
}

core::Status Sqrt(const core::Context &ctx, const core::Tensor &x,
            core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatch(x, result, [](const auto *px, auto *po, size_t n) {
    UnaryKernel(px, po, n, [](auto v) { return std::sqrt(v); });
  });
}

core::Status Sigmoid(const core::Context &ctx, const core::Tensor &x,
               core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatch(x, result, [](const auto *px, auto *po, size_t n) {
    UnaryKernel(px, po, n, [](auto v) {
      using V = decltype(v);
      return V(1) / (V(1) + std::exp(-v));
    });
  });
}

core::Status Tanh(const core::Context &ctx, const core::Tensor &x,
            core::Tensor &result) {
  (void)ctx;
  core::Status st = UnaryOpCheck(x, result);
  if (st != core::Status::OK) return st;
  return UnaryDispatch(x, result, [](const auto *px, auto *po, size_t n) {
    UnaryKernel(px, po, n, [](auto v) { return std::tanh(v); });
  });
}

}  // namespace cpu
}  // namespace ops
}  // namespace axono
