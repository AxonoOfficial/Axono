// LLM 相关算子 (CPU): softmax/log_softmax/gelu/silu/layernorm/rmsnorm。
// 沿最后一维归约, 支持任意前置维; 行内 OpenMP (行数多时按行并行),
// 仅浮点 dtype。
#include <cmath>
#include <cstddef>
#include <vector>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include "axono/ops/cpu/llm.h"

namespace axono {
namespace ops {
namespace cpu {

namespace {

// 展开最后一维: rows = prod(shape[:-1]), cols = shape[-1]
core::Status LastDimCheck(const core::Tensor &x, core::Tensor &result,
                          size_t &rows, size_t &cols) {
  if (x.ndim() < 1) return core::Status::INVALID_ARGUMENT;
  const auto &shape = x.shape();
  cols = shape.back();
  rows = x.num_elements() / cols;
  core::Status st = result.Resize(shape);
  if (st != core::Status::OK) return st;
  if (result.dtype() != x.dtype()) return core::Status::UNSUPPORTED_TYPE;
  if (cols == 0) return core::Status::INVALID_ARGUMENT;
  return core::Status::OK;
}

template <typename T>
void SoftmaxRows(const T *x, T *out, size_t rows, size_t cols,
                 bool log_mode) {
#pragma omp parallel for schedule(static)
  for (size_t r = 0; r < rows; ++r) {
    const T *xr = x + r * cols;
    T *orr = out + r * cols;
    T mx = xr[0];
    for (size_t c = 1; c < cols; ++c) mx = xr[c] > mx ? xr[c] : mx;
    T sum = T(0);
    for (size_t c = 0; c < cols; ++c) {
      orr[c] = std::exp(xr[c] - mx);
      sum += orr[c];
    }
    if (log_mode) {
      const T lse = std::log(sum) + mx;
      for (size_t c = 0; c < cols; ++c) orr[c] = xr[c] - lse;
    } else {
      const T inv = T(1) / sum;
      for (size_t c = 0; c < cols; ++c) orr[c] *= inv;
    }
  }
}

template <typename T>
void NormRows(const T *x, T *out, size_t rows, size_t cols, const T *weight,
              const T *bias, bool has_weight, bool has_bias, float eps,
              bool rms_mode) {
#pragma omp parallel for schedule(static)
  for (size_t r = 0; r < rows; ++r) {
    const T *xr = x + r * cols;
    T *orr = out + r * cols;
    if (rms_mode) {
      T ss = T(0);
      for (size_t c = 0; c < cols; ++c) ss += xr[c] * xr[c];
      const T inv = T(1) / std::sqrt(ss / static_cast<T>(cols) +
                                     static_cast<T>(eps));
      for (size_t c = 0; c < cols; ++c) {
        const T normed = xr[c] * inv;
        orr[c] = has_weight ? normed * weight[c] : normed;
      }
    } else {
      T mean = T(0);
      for (size_t c = 0; c < cols; ++c) mean += xr[c];
      mean /= static_cast<T>(cols);
      T var = T(0);
      for (size_t c = 0; c < cols; ++c) {
        const T d = xr[c] - mean;
        var += d * d;
      }
      var /= static_cast<T>(cols);
      const T inv = T(1) / std::sqrt(var + static_cast<T>(eps));
      for (size_t c = 0; c < cols; ++c) {
        const T normed = (xr[c] - mean) * inv;
        orr[c] = (has_weight ? normed * weight[c] : normed) +
                 (has_bias ? bias[c] : T(0));
      }
    }
  }
}

}  // namespace

core::Status Softmax(const core::Context &ctx, const core::Tensor &x,
                     core::Tensor &result) {
  (void)ctx;
  size_t rows = 0, cols = 0;
  core::Status st = LastDimCheck(x, result, rows, cols);
  if (st != core::Status::OK) return st;
  if (x.dtype() == core::DataType::FLOAT32)
    SoftmaxRows<float>(x.data<float>(), result.data<float>(), rows, cols,
                       false);
  else if (x.dtype() == core::DataType::FLOAT64)
    SoftmaxRows<double>(x.data<double>(), result.data<double>(), rows, cols,
                        false);
  else
    return core::Status::UNSUPPORTED_TYPE;
  return core::Status::OK;
}

core::Status LogSoftmax(const core::Context &ctx, const core::Tensor &x,
                        core::Tensor &result) {
  (void)ctx;
  size_t rows = 0, cols = 0;
  core::Status st = LastDimCheck(x, result, rows, cols);
  if (st != core::Status::OK) return st;
  if (x.dtype() == core::DataType::FLOAT32)
    SoftmaxRows<float>(x.data<float>(), result.data<float>(), rows, cols, true);
  else if (x.dtype() == core::DataType::FLOAT64)
    SoftmaxRows<double>(x.data<double>(), result.data<double>(), rows, cols,
                        true);
  else
    return core::Status::UNSUPPORTED_TYPE;
  return core::Status::OK;
}

core::Status Gelu(const core::Context &ctx, const core::Tensor &x,
                  core::Tensor &result) {
  (void)ctx;
  size_t rows = 0, cols = 0;
  core::Status st = LastDimCheck(x, result, rows, cols);
  if (st != core::Status::OK) return st;
  const size_t n = x.num_elements();
  if (x.dtype() == core::DataType::FLOAT32) {
    const float *px = x.data<float>();
    float *po = result.data<float>();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
      po[i] = 0.5f * px[i] * (1.0f + std::erf(px[i] / 1.41421356237f));
  } else if (x.dtype() == core::DataType::FLOAT64) {
    const double *px = x.data<double>();
    double *po = result.data<double>();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
      po[i] = 0.5 * px[i] * (1.0 + std::erf(px[i] / 1.4142135623730951));
  } else {
    return core::Status::UNSUPPORTED_TYPE;
  }
  return core::Status::OK;
}

// GELU (tanh 近似式): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3)))。
core::Status GeluTanh(const core::Context &ctx, const core::Tensor &x,
                      core::Tensor &result) {
  (void)ctx;
  size_t rows = 0, cols = 0;
  core::Status st = LastDimCheck(x, result, rows, cols);
  if (st != core::Status::OK) return st;
  const size_t n = x.num_elements();
  if (x.dtype() == core::DataType::FLOAT32) {
    const float *px = x.data<float>();
    float *po = result.data<float>();
    constexpr float kC = 0.7978845608028654f;  // sqrt(2/pi)
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      const float v = px[i];
      const float inner = kC * (v + 0.044715f * v * v * v);
      po[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
  } else if (x.dtype() == core::DataType::FLOAT64) {
    const double *px = x.data<double>();
    double *po = result.data<double>();
    constexpr double kC = 0.7978845608028654;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      const double v = px[i];
      const double inner = kC * (v + 0.044715 * v * v * v);
      po[i] = 0.5 * v * (1.0 + std::tanh(inner));
    }
  } else {
    return core::Status::UNSUPPORTED_TYPE;
  }
  return core::Status::OK;
}

core::Status Silu(const core::Context &ctx, const core::Tensor &x,
                  core::Tensor &result) {
  (void)ctx;
  size_t rows = 0, cols = 0;
  core::Status st = LastDimCheck(x, result, rows, cols);
  if (st != core::Status::OK) return st;
  const size_t n = x.num_elements();
  if (x.dtype() == core::DataType::FLOAT32) {
    const float *px = x.data<float>();
    float *po = result.data<float>();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
      po[i] = px[i] / (1.0f + std::exp(-px[i]));
  } else if (x.dtype() == core::DataType::FLOAT64) {
    const double *px = x.data<double>();
    double *po = result.data<double>();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
      po[i] = px[i] / (1.0 + std::exp(-px[i]));
  } else {
    return core::Status::UNSUPPORTED_TYPE;
  }
  return core::Status::OK;
}

core::Status LayerNorm(const core::Context &ctx, const core::Tensor &x,
                       const core::Tensor &weight, const core::Tensor &bias,
                       float eps, core::Tensor &result) {
  (void)ctx;
  size_t rows = 0, cols = 0;
  core::Status st = LastDimCheck(x, result, rows, cols);
  if (st != core::Status::OK) return st;
  const bool has_w = weight.num_elements() != 0;
  const bool has_b = bias.num_elements() != 0;
  if ((has_w && (weight.ndim() != 1 || weight.shape()[0] != cols ||
                 weight.dtype() != x.dtype())) ||
      (has_b && (bias.ndim() != 1 || bias.shape()[0] != cols ||
                 bias.dtype() != x.dtype())))
    return core::Status::SHAPE_MISMATCH;
  if (x.dtype() == core::DataType::FLOAT32)
    NormRows<float>(x.data<float>(), result.data<float>(), rows, cols,
                    has_w ? weight.data<float>() : nullptr,
                    has_b ? bias.data<float>() : nullptr, has_w, has_b, eps,
                    false);
  else if (x.dtype() == core::DataType::FLOAT64)
    NormRows<double>(x.data<double>(), result.data<double>(), rows, cols,
                     has_w ? weight.data<double>() : nullptr,
                     has_b ? bias.data<double>() : nullptr, has_w, has_b, eps,
                     false);
  else
    return core::Status::UNSUPPORTED_TYPE;
  return core::Status::OK;
}

core::Status RmsNorm(const core::Context &ctx, const core::Tensor &x,
                     const core::Tensor &weight, float eps,
                     core::Tensor &result) {
  (void)ctx;
  size_t rows = 0, cols = 0;
  core::Status st = LastDimCheck(x, result, rows, cols);
  if (st != core::Status::OK) return st;
  const bool has_w = weight.num_elements() != 0;
  if (has_w && (weight.ndim() != 1 || weight.shape()[0] != cols ||
                weight.dtype() != x.dtype()))
    return core::Status::SHAPE_MISMATCH;
  core::Tensor empty_bias;
  if (x.dtype() == core::DataType::FLOAT32)
    NormRows<float>(x.data<float>(), result.data<float>(), rows, cols,
                    has_w ? weight.data<float>() : nullptr, nullptr, has_w,
                    false, eps, true);
  else if (x.dtype() == core::DataType::FLOAT64)
    NormRows<double>(x.data<double>(), result.data<double>(), rows, cols,
                     has_w ? weight.data<double>() : nullptr, nullptr, has_w,
                     false, eps, true);
  else
    return core::Status::UNSUPPORTED_TYPE;
  (void)empty_bias;
  return core::Status::OK;
}

}  // namespace cpu
}  // namespace ops
}  // namespace axono
