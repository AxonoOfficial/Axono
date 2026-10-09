// 单算子融合 Linear (CPU): result = x @ W^T + bias。
// BLAS 可用时 cblas gemm (beta=1, 先把 bias 广播进 result),
// 否则分块朴素实现; 仅浮点 dtype。
// 支持任意前置维 batch: x (..., k) -> result (..., out_f)。
#include <cstddef>
#include <vector>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include "axono/ops/cpu/linear.h"

#ifdef AXONO_WITH_BLAS
#include <cblas.h>
#endif

namespace axono {
namespace ops {
namespace cpu {

namespace {

core::Status LinearCheck(const core::Tensor &x, const core::Tensor &weight,
                         const core::Tensor &bias, core::Tensor &result,
                         size_t &rows, size_t &k, size_t &out_f) {
  if (x.ndim() < 1 || weight.ndim() != 2)
    return core::Status::INVALID_ARGUMENT;
  k = x.shape().back();
  out_f = weight.shape()[0];
  rows = x.num_elements() / k;
  if (weight.shape()[1] != k) return core::Status::SHAPE_MISMATCH;
  if (bias.num_elements() != 0 &&
      (bias.ndim() != 1 || bias.shape()[0] != out_f))
    return core::Status::SHAPE_MISMATCH;
  if (x.dtype() != weight.dtype() || x.dtype() != result.dtype())
    return core::Status::UNSUPPORTED_TYPE;
  if (bias.num_elements() != 0 && bias.dtype() != x.dtype())
    return core::Status::UNSUPPORTED_TYPE;
  std::vector<size_t> out_shape = x.shape();
  out_shape.back() = out_f;
  core::Status st = result.Resize(out_shape);
  if (st != core::Status::OK) return st;
  return core::Status::OK;
}

}  // namespace

core::Status Linear(const core::Context &ctx, const core::Tensor &x,
                    const core::Tensor &weight, const core::Tensor &bias,
                    core::Tensor &result) {
  (void)ctx;
  size_t rows = 0, k = 0, out_f = 0;
  core::Status st = LinearCheck(x, weight, bias, result, rows, k, out_f);
  if (st != core::Status::OK) return st;
  const bool has_bias = bias.num_elements() != 0;

  // 先把 bias 广播进 result (作为 gemm 的 C 初值), 无 bias 则清零
  if (has_bias) {
    if (x.dtype() == core::DataType::FLOAT32) {
      float *rf = result.data<float>();
      const float *bf = bias.data<float>();
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < out_f; ++j) rf[i * out_f + j] = bf[j];
    } else {
      double *rd = result.data<double>();
      const double *bd = bias.data<double>();
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < out_f; ++j) rd[i * out_f + j] = bd[j];
    }
  } else {
    core::Status st2 = result.FillZero();
    if (st2 != core::Status::OK) return st2;
  }

  if (rows == 0 || k == 0 || out_f == 0) return core::Status::OK;

#ifdef AXONO_WITH_BLAS
  if (x.dtype() == core::DataType::FLOAT32) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(rows),
                static_cast<int>(out_f), static_cast<int>(k), 1.0f,
                x.data<float>(), static_cast<int>(k), weight.data<float>(),
                static_cast<int>(k), 1.0f, result.data<float>(),
                static_cast<int>(out_f));
    return core::Status::OK;
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(rows),
              static_cast<int>(out_f), static_cast<int>(k), 1.0,
              x.data<double>(), static_cast<int>(k), weight.data<double>(),
              static_cast<int>(k), 1.0, result.data<double>(),
              static_cast<int>(out_f));
  return core::Status::OK;
#else
  if (x.dtype() == core::DataType::FLOAT32) {
    const float *px = x.data<float>();
    const float *pw = weight.data<float>();
    float *po = result.data<float>();
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < out_f; ++j) {
        float acc = 0.0f;
        for (size_t p = 0; p < k; ++p) acc += px[i * k + p] * pw[j * k + p];
        po[i * out_f + j] = acc;
      }
    }
    return core::Status::OK;
  }
  const double *px = x.data<double>();
  const double *pw = weight.data<double>();
  double *po = result.data<double>();
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < out_f; ++j) {
      double acc = 0.0;
      for (size_t p = 0; p < k; ++p) acc += px[i * k + p] * pw[j * k + p];
      po[i * out_f + j] = acc;
    }
  }
  return core::Status::OK;
#endif
}

}  // namespace cpu
}  // namespace ops
}  // namespace axono
