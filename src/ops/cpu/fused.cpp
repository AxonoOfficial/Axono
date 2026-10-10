// 融合算子 (CPU): 语义同 cuda 侧 SiluMul / AddRmsNorm。
#include "axono/ops/cpu/fused.h"

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "axono/core/tensor.h"

namespace axono {
namespace ops {
namespace cpu {

core::Status SiluMul(const core::Context &ctx, const core::Tensor &gate,
                     const core::Tensor &up, core::Tensor &result) {
  (void)ctx;
  if (gate.dtype() != core::DataType::FLOAT32 ||
      gate.shape() != up.shape())
    return core::Status::SHAPE_MISMATCH;
  core::Status st = result.Resize(gate.shape());
  if (st != core::Status::OK) return st;
  const size_t n = gate.num_elements();
  const float *gp = gate.data<float>();
  const float *upp = up.data<float>();
  float *yp = result.data<float>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < n; ++i) {
    const float g = gp[i];
    yp[i] = g / (1.0f + std::exp(-g)) * upp[i];
  }
  return core::Status::OK;
}

core::Status AddRmsNorm(const core::Context &ctx, const core::Tensor &x,
                        const core::Tensor &residual,
                        const core::Tensor &weight, float eps,
                        core::Tensor &y, core::Tensor &out) {
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
  const float *xp = x.data<float>();
  const float *rp = residual.data<float>();
  const float *wp = weight.data<float>();
  float *yp = y.data<float>();
  float *op = out.data<float>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t r = 0; r < rows; ++r) {
    const float *xr = xp + r * hidden;
    const float *rr = rp + r * hidden;
    float *yr = yp + r * hidden;
    float *orr = op + r * hidden;
    float sum = 0.0f;
    for (size_t i = 0; i < hidden; ++i) {
      const float v = xr[i] + rr[i];
      yr[i] = v;
      sum += v * v;
    }
    const float rms = 1.0f / std::sqrt(sum / static_cast<float>(hidden) + eps);
    for (size_t i = 0; i < hidden; ++i) orr[i] = yr[i] * rms * wp[i];
  }
  return core::Status::OK;
}

}  // namespace cpu
}  // namespace ops
}  // namespace axono
