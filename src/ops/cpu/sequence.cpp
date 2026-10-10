// 序列/LLM 结构算子 (CPU): embedding/rope/scaled_dot_product_attention/
// concat/slice/argmax_last_dim。仅浮点 (attention/rope 仅 FLOAT32)。
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include "axono/ops/cpu/sequence.h"

namespace axono {
namespace ops {
namespace cpu {

core::Status Embedding(const core::Context &ctx, const core::Tensor &ids,
                       const core::Tensor &table, core::Tensor &result) {
  (void)ctx;
  if (ids.ndim() < 1 || table.ndim() != 2)
    return core::Status::INVALID_ARGUMENT;
  if (table.dtype() != core::DataType::FLOAT32)
    return core::Status::UNSUPPORTED_TYPE;
  const size_t hidden = table.shape()[1];
  std::vector<size_t> out_shape = ids.shape();
  out_shape.push_back(hidden);
  core::Status st = result.Resize(out_shape);
  if (st != core::Status::OK) return st;
  if (result.dtype() != core::DataType::FLOAT32)
    return core::Status::UNSUPPORTED_TYPE;

  const float *tp = table.data<float>();
  float *op = result.data<float>();
  const size_t n = ids.num_elements();
  if (ids.dtype() == core::DataType::INT32) {
    const int32_t *ip = ids.data<int32_t>();
    for (size_t i = 0; i < n; ++i) {
      const size_t id = static_cast<size_t>(ip[i]);
      std::memcpy(op + i * hidden, tp + id * hidden, hidden * sizeof(float));
    }
  } else if (ids.dtype() == core::DataType::INT64) {
    const int64_t *ip = ids.data<int64_t>();
    for (size_t i = 0; i < n; ++i) {
      const size_t id = static_cast<size_t>(ip[i]);
      std::memcpy(op + i * hidden, tp + id * hidden, hidden * sizeof(float));
    }
  } else {
    return core::Status::UNSUPPORTED_TYPE;
  }
  return core::Status::OK;
}

core::Status Rope(const core::Context &ctx, const core::Tensor &x,
                  const core::Tensor &pos_ids, float theta,
                  core::Tensor &result) {
  (void)ctx;
  if (x.ndim() != 3 || x.dtype() != core::DataType::FLOAT32)
    return core::Status::UNSUPPORTED_TYPE;
  const size_t rows = x.shape()[0];
  const size_t heads = x.shape()[1];
  const size_t dim = x.shape()[2];
  if (dim % 2 != 0) return core::Status::INVALID_ARGUMENT;
  if (pos_ids.num_elements() != rows || pos_ids.dtype() != core::DataType::INT64)
    return core::Status::UNSUPPORTED_TYPE;
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;

  const float *xp = x.data<float>();
  float *rp = result.data<float>();
  const int64_t *pos = pos_ids.data<int64_t>();
  const size_t half = dim / 2;
  // HF rotate_half 约定: 配对 (i, i+half), i in [0, half)
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t r = 0; r < rows; ++r) {
    for (size_t h = 0; h < heads; ++h) {
      const float *src = xp + (r * heads + h) * dim;
      float *dst = rp + (r * heads + h) * dim;
      const float p = static_cast<float>(pos[r]);
      for (size_t i = 0; i < half; ++i) {
        const float freq =
            std::pow(theta, -2.0f * static_cast<float>(i) / static_cast<float>(dim));
        const float ang = p * freq;
        const float c = std::cos(ang), s = std::sin(ang);
        const float a = src[i], b = src[i + half];
        dst[i] = a * c - b * s;
        dst[i + half] = b * c + a * s;
      }
    }
  }
  return core::Status::OK;
}

// RoPE 变体 1: 直接 (cos, sin) 旋转 (M-RoPE / 3D 位置)。
core::Status RopeWithCosSin(const core::Context &ctx, const core::Tensor &x,
                            const core::Tensor &cos, const core::Tensor &sin,
                            core::Tensor &result) {
  (void)ctx;
  if (x.ndim() != 3 || x.dtype() != core::DataType::FLOAT32)
    return core::Status::UNSUPPORTED_TYPE;
  const size_t rows = x.shape()[0];
  const size_t heads = x.shape()[1];
  const size_t dim = x.shape()[2];
  if (dim % 2 != 0) return core::Status::INVALID_ARGUMENT;
  if (cos.ndim() != 2 || sin.ndim() != 2 || cos.shape()[0] != rows ||
      cos.shape()[1] != dim || sin.shape()[0] != rows ||
      sin.shape()[1] != dim || cos.dtype() != core::DataType::FLOAT32 ||
      sin.dtype() != core::DataType::FLOAT32)
    return core::Status::SHAPE_MISMATCH;
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;

  const float *xp = x.data<float>();
  float *rp = result.data<float>();
  const float *cp = cos.data<float>();
  const float *sp = sin.data<float>();
  const size_t half = dim / 2;
  // HF rotate_half 约定: 配对 (i, i+half), i in [0, half)。
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t r = 0; r < rows; ++r) {
    for (size_t h = 0; h < heads; ++h) {
      const float *src = xp + (r * heads + h) * dim;
      float *dst = rp + (r * heads + h) * dim;
      const float *cr = cp + r * dim;
      const float *sr = sp + r * dim;
      for (size_t i = 0; i < half; ++i) {
        const float a = src[i], b = src[i + half];
        dst[i] = a * cr[i] - b * sr[i];
        dst[i + half] = b * cr[i + half] + a * sr[i + half];
      }
    }
  }
  return core::Status::OK;
}

// RoPE 变体 2: 3D 位置 (T/H/W) 交错 (stride-3) 频率重组。
// 逐 (r,h,i): freq_i 取 T 基础, i%3==1 且 i<3*h_sec 用 H, i%3==2 且 i<3*w_sec 用 W。
core::Status MropeCosSin(const core::Context &ctx, const core::Tensor &pos,
                         const core::Tensor &inv_freq, int h_sec, int w_sec,
                         core::Tensor &cos_out, core::Tensor &sin_out) {
  (void)ctx;
  if (pos.ndim() != 2 || pos.shape()[0] != 3 || pos.dtype() != core::DataType::INT64)
    return core::Status::SHAPE_MISMATCH;
  const size_t rows = pos.shape()[1];
  const size_t half = inv_freq.shape()[0];
  if (half == 0) return core::Status::INVALID_ARGUMENT;
  core::Status st = cos_out.Resize({rows, 2 * half});
  if (st != core::Status::OK) return st;
  st = sin_out.Resize({rows, 2 * half});
  if (st != core::Status::OK) return st;
  const int64_t *pp = pos.data<int64_t>();
  const float *fp = inv_freq.data<float>();
  float *cp = cos_out.data<float>();
  float *sp = sin_out.data<float>();
  const size_t h_lim = static_cast<size_t>(h_sec) * 3;
  const size_t w_lim = static_cast<size_t>(w_sec) * 3;
#pragma omp parallel for schedule(static)
  for (size_t r = 0; r < rows; ++r) {
    const float pt = static_cast<float>(pp[r]);
    const float ph = static_cast<float>(pp[rows + r]);
    const float pw = static_cast<float>(pp[2 * rows + r]);
    float *cr = cp + r * (2 * half);
    float *sr = sp + r * (2 * half);
    for (size_t i = 0; i < half; ++i) {
      const float f = fp[i];
      float angle = pt * f;
      if (i % 3 == 1 && i < h_lim)
        angle = ph * f;
      else if (i % 3 == 2 && i < w_lim)
        angle = pw * f;
      const float c = std::cos(angle), sn = std::sin(angle);
      cr[i] = c;
      cr[i + half] = c;
      sr[i] = sn;
      sr[i + half] = sn;
    }
  }
  return core::Status::OK;
}

core::Status RopeThd(const core::Context &ctx, const core::Tensor &x,
                     const core::Tensor &pos, const core::Tensor &inv_freq,
                     int t_sec, int h_sec, int w_sec, core::Tensor &result) {
  (void)ctx;
  if (x.ndim() != 3 || x.dtype() != core::DataType::FLOAT32)
    return core::Status::UNSUPPORTED_TYPE;
  const size_t rows = x.shape()[0];
  const size_t heads = x.shape()[1];
  const size_t dim = x.shape()[2];
  if (dim % 2 != 0) return core::Status::INVALID_ARGUMENT;
  const size_t half = dim / 2;
  if (pos.ndim() != 2 || pos.shape()[0] != 3 || pos.shape()[1] != rows ||
      pos.dtype() != core::DataType::INT64)
    return core::Status::SHAPE_MISMATCH;
  if (inv_freq.ndim() != 1 || inv_freq.shape()[0] != half ||
      inv_freq.dtype() != core::DataType::FLOAT32)
    return core::Status::SHAPE_MISMATCH;
  (void)t_sec;
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;

  const float *xp = x.data<float>();
  float *rp = result.data<float>();
  const int64_t *pp = pos.data<int64_t>();
  const float *fp = inv_freq.data<float>();
  const size_t h_lim = static_cast<size_t>(h_sec) * 3;
  const size_t w_lim = static_cast<size_t>(w_sec) * 3;
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t r = 0; r < rows; ++r) {
    for (size_t h = 0; h < heads; ++h) {
      const float *src = xp + (r * heads + h) * dim;
      float *dst = rp + (r * heads + h) * dim;
      const float pt = static_cast<float>(pp[r]);
      const float ph = static_cast<float>(pp[rows + r]);
      const float pw = static_cast<float>(pp[2 * rows + r]);
      for (size_t i = 0; i < half; ++i) {
        const float f = fp[i];
        float angle = pt * f;
        if (i % 3 == 1 && i < h_lim)
          angle = ph * f;
        else if (i % 3 == 2 && i < w_lim)
          angle = pw * f;
        const float c = std::cos(angle), s = std::sin(angle);
        const float a = src[i], b = src[i + half];
        dst[i] = a * c - b * s;
        dst[i + half] = b * c + a * s;
      }
    }
  }
  return core::Status::OK;
}

core::Status ScaledDotProductAttention(const core::Context &ctx,
                                       const core::Tensor &q,
                                       const core::Tensor &k,
                                       const core::Tensor &v, bool is_causal,
                                       core::Tensor &result) {
  (void)ctx;
  if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3)
    return core::Status::INVALID_ARGUMENT;
  if (q.dtype() != core::DataType::FLOAT32 ||
      k.dtype() != core::DataType::FLOAT32 ||
      v.dtype() != core::DataType::FLOAT32)
    return core::Status::UNSUPPORTED_TYPE;
  const size_t sq = q.shape()[0], hq = q.shape()[1], d = q.shape()[2];
  const size_t skv = k.shape()[0], hkv = k.shape()[1];
  if (k.shape()[2] != d || v.shape()[0] != skv || v.shape()[1] != hkv ||
      v.shape()[2] != d)
    return core::Status::SHAPE_MISMATCH;
  if (hq % hkv != 0) return core::Status::SHAPE_MISMATCH;
  const size_t group = hq / hkv;
  core::Status st = result.Resize({sq, hq, d});
  if (st != core::Status::OK) return st;

  const float *qp = q.data<float>();
  const float *kp = k.data<float>();
  const float *vp = v.data<float>();
  float *rp = result.data<float>();
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

#pragma omp parallel for collapse(2) schedule(static)
  for (size_t t = 0; t < sq; ++t) {
    for (size_t h = 0; h < hq; ++h) {
      const float *qr = qp + (t * hq + h) * d;
      float *outr = rp + (t * hq + h) * d;
      const size_t hk = h / group;
      // 可见的 key 范围: causal 时 [0, t] (前提 skv >= t+1), 否则全部
      size_t kv_end = skv;
      if (is_causal && skv >= t + 1) kv_end = t + 1;

      // scores = Q @ K^T * scale (kv_end 个)
      std::vector<float> scores(kv_end);
      for (size_t s = 0; s < kv_end; ++s) {
        const float *kr = kp + (s * hkv + hk) * d;
        float acc = 0.0f;
        for (size_t i = 0; i < d; ++i) acc += qr[i] * kr[i];
        scores[s] = acc * scale;
      }
      // 数值稳定 softmax
      float mx = scores[0];
      for (size_t s = 1; s < kv_end; ++s) mx = scores[s] > mx ? scores[s] : mx;
      float sum = 0.0f;
      for (size_t s = 0; s < kv_end; ++s) {
        scores[s] = std::exp(scores[s] - mx);
        sum += scores[s];
      }
      const float inv = 1.0f / sum;
      // out = sum_s p[s] * V[s]
      for (size_t i = 0; i < d; ++i) outr[i] = 0.0f;
      for (size_t s = 0; s < kv_end; ++s) {
        const float p = scores[s] * inv;
        const float *vr = vp + (s * hkv + hk) * d;
        for (size_t i = 0; i < d; ++i) outr[i] += p * vr[i];
      }
    }
  }
  return core::Status::OK;
}

core::Status Concat(const core::Context &ctx, const core::Tensor &a,
                    const core::Tensor &b, int axis, core::Tensor &result) {
  (void)ctx;
  if (a.ndim() < 1 || a.ndim() != b.ndim() || a.dtype() != b.dtype())
    return core::Status::INVALID_ARGUMENT;
  const size_t nd = a.ndim();
  size_t ax = axis < 0 ? nd + static_cast<size_t>(axis) : static_cast<size_t>(axis);
  if (ax >= nd) return core::Status::INVALID_ARGUMENT;
  for (size_t i = 0; i < nd; ++i) {
    if (i != ax && a.shape()[i] != b.shape()[i])
      return core::Status::SHAPE_MISMATCH;
  }
  std::vector<size_t> shape = a.shape();
  shape[ax] += b.shape()[ax];
  core::Status st = result.Resize(shape);
  if (st != core::Status::OK) return st;

  // 按最外层 (axis 之前) 的块拷贝: outer = prod(shape[:axis]), inner 之后连续
  size_t outer = 1;
  for (size_t i = 0; i < ax; ++i) outer *= a.shape()[i];
  const size_t a_inner = a.num_elements() / outer;   // 每 outer 块元素数
  const size_t b_inner = b.num_elements() / outer;
  const size_t a_bytes = a_inner * core::GetDataTypeSize(a.dtype());
  const size_t b_bytes = b_inner * core::GetDataTypeSize(b.dtype());
  const char *ap = static_cast<const char *>(a.data());
  const char *bp = static_cast<const char *>(b.data());
  char *rp = static_cast<char *>(result.data());
  for (size_t o = 0; o < outer; ++o) {
    std::memcpy(rp, ap + o * a_bytes, a_bytes);
    rp += a_bytes;
    std::memcpy(rp, bp + o * b_bytes, b_bytes);
    rp += b_bytes;
  }
  return core::Status::OK;
}

core::Status Slice(const core::Context &ctx, const core::Tensor &x,
                   size_t axis, size_t start, size_t length,
                   core::Tensor &result) {
  (void)ctx;
  if (x.ndim() < 1) return core::Status::INVALID_ARGUMENT;
  if (axis >= x.ndim() || start + length > x.shape()[axis])
    return core::Status::INVALID_ARGUMENT;
  std::vector<size_t> shape = x.shape();
  shape[axis] = length;
  core::Status st = result.Resize(shape);
  if (st != core::Status::OK) return st;
  if (x.dtype() != result.dtype()) return core::Status::UNSUPPORTED_TYPE;

  size_t outer = 1;
  for (size_t i = 0; i < axis; ++i) outer *= x.shape()[i];
  const size_t row = x.num_elements() / x.shape()[axis] /
                     outer;  // axis 之后每段元素数
  const size_t src_len = x.shape()[axis];
  const size_t seg = row * core::GetDataTypeSize(x.dtype());
  const char *xp = static_cast<const char *>(x.data());
  char *rp = static_cast<char *>(result.data());
  for (size_t o = 0; o < outer; ++o) {
    std::memcpy(rp, xp + ((o * src_len) + start) * seg, length * seg);
    rp += length * seg;
  }
  return core::Status::OK;
}

core::Status ArgmaxLastDim(const core::Context &ctx, const core::Tensor &x,
                           core::Tensor &result) {
  (void)ctx;
  if (x.ndim() < 1) return core::Status::INVALID_ARGUMENT;
  const size_t cols = x.shape().back();
  const size_t rows = x.num_elements() / cols;
  core::Status st = result.Resize({rows});
  if (st != core::Status::OK) return st;
  if (result.dtype() != core::DataType::INT64)
    return core::Status::UNSUPPORTED_TYPE;
  int64_t *rp = result.data<int64_t>();
  if (x.dtype() == core::DataType::FLOAT32) {
    const float *xp = x.data<float>();
    for (size_t r = 0; r < rows; ++r) {
      const float *row = xp + r * cols;
      int64_t best = 0;
      for (size_t c = 1; c < cols; ++c)
        if (row[c] > row[best]) best = static_cast<int64_t>(c);
      rp[r] = best;
    }
  } else if (x.dtype() == core::DataType::FLOAT64) {
    const double *xp = x.data<double>();
    for (size_t r = 0; r < rows; ++r) {
      const double *row = xp + r * cols;
      int64_t best = 0;
      for (size_t c = 1; c < cols; ++c)
        if (row[c] > row[best]) best = static_cast<int64_t>(c);
      rp[r] = best;
    }
  } else {
    return core::Status::UNSUPPORTED_TYPE;
  }
  return core::Status::OK;
}

}  // namespace cpu
}  // namespace ops
}  // namespace axono
