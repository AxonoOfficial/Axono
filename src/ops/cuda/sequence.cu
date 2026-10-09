// 序列/LLM 结构算子 (CUDA): embedding/rope/scaled_dot_product_attention/
// concat/slice/argmax_last_dim。仅浮点; 兼容 Graph 捕获。
#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

#include "axono/core/cuda/capture.h"
#include "axono/core/cuda/stream.h"
#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include "axono/ops/cuda/sequence.h"

namespace axono {
namespace ops {
namespace cuda {

namespace {

constexpr int kBlock = 256;

inline cudaStream_t Stream() { return core::cuda::AxonoCurrentStream(); }

inline core::Status Finish() {
  if (cudaGetLastError() != cudaSuccess) return core::Status::DEVICE_ERROR;
  if (core::cuda::IsCapturing()) return core::Status::OK;
  return cudaDeviceSynchronize() == cudaSuccess ? core::Status::OK
                                                : core::Status::INTERNAL_ERROR;
}

__global__ void EmbeddingKernel(const int64_t *ids, const float *table,
                                float *out, size_t n, size_t hidden) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n * hidden) return;
  const size_t tok = idx / hidden;
  out[idx] = table[static_cast<size_t>(ids[tok]) * hidden + idx % hidden];
}

// RoPE (HF rotate_half 约定: 配对 (i, i+dim/2)): 每 (pos, head) 一个 block
__global__ void RopeKernel(float *x, const int64_t *pos, size_t rows,
                           size_t heads, size_t dim, float theta) {
  const size_t r = blockIdx.x;
  const size_t h = blockIdx.y;
  if (r >= rows || h >= heads) return;
  float *d = x + (r * heads + h) * dim;
  const float p = static_cast<float>(pos[r]);
  const size_t half = dim / 2;
  for (size_t i = threadIdx.x; i < half; i += blockDim.x) {
    const float freq = powf(theta, -2.0f * static_cast<float>(i) /
                                       static_cast<float>(dim));
    const float ang = p * freq;
    const float c = cosf(ang), s = sinf(ang);
    const float a = d[i], b = d[i + half];
    d[i] = a * c - b * s;
    d[i + half] = b * c + a * s;
  }
}

// SDPA: 每 (t, head) 一个线程块, block 内对 kv 求分数 + softmax + 加权 V
__global__ void SdpaKernel(const float *q, const float *k, const float *v,
                           float *out, size_t sq, size_t hq, size_t skv,
                           size_t hkv, size_t d, int causal) {
  const size_t t = blockIdx.x;
  const size_t h = blockIdx.y;
  if (t >= sq || h >= hq) return;
  const size_t group = hq / hkv;
  const size_t hk = h / group;
  const float *qr = q + (t * hq + h) * d;
  float *outr = out + (t * hq + h) * d;
  const float scale = rsqrtf(static_cast<float>(d));

  size_t kv_end = skv;
  if (causal && skv >= t + 1) kv_end = t + 1;

  // 单线程算分数+softmax (解码场景 kv 短, 简单可靠; prefill 也正确)
  if (threadIdx.x == 0) {
    extern __shared__ float scores[];
    float mx = -3.4e38f;
    for (size_t s = 0; s < kv_end; ++s) {
      const float *kr = k + (s * hkv + hk) * d;
      float acc = 0.0f;
      for (size_t i = 0; i < d; ++i) acc += qr[i] * kr[i];
      scores[s] = acc * scale;
      if (scores[s] > mx) mx = scores[s];
    }
    float sum = 0.0f;
    for (size_t s = 0; s < kv_end; ++s) {
      scores[s] = expf(scores[s] - mx);
      sum += scores[s];
    }
    for (size_t i = 0; i < d; ++i) outr[i] = 0.0f;
    for (size_t s = 0; s < kv_end; ++s) {
      const float p = scores[s] / sum;
      const float *vr = v + (s * hkv + hk) * d;
      for (size_t i = 0; i < d; ++i) outr[i] += p * vr[i];
    }
  }
}

// Concat axis=0: 直接两段拷贝; axis=-1 (最后一维): 每 (outer, idx) 重排
__global__ void ConcatLastDimKernel(const char *a, const char *b, char *out,
                                    size_t outer, size_t a_cols, size_t b_cols,
                                    size_t elem) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t cols = a_cols + b_cols;
  if (idx >= outer * cols) return;
  const size_t o = idx / cols;
  const size_t c = idx % cols;
  if (c < a_cols) {
    for (size_t e = 0; e < elem; ++e)
      out[idx * elem + e] = a[(o * a_cols + c) * elem + e];
  } else {
    const size_t cb = c - a_cols;
    for (size_t e = 0; e < elem; ++e)
      out[idx * elem + e] = b[(o * b_cols + cb) * elem + e];
  }
}

// Slice 最后一维 [start, start+len)
__global__ void SliceLastDimKernel(const char *x, char *out, size_t outer,
                                   size_t src_cols, size_t start, size_t len,
                                   size_t elem) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= outer * len) return;
  const size_t o = idx / len;
  const size_t c = idx % len;
  for (size_t e = 0; e < elem; ++e)
    out[idx * elem + e] = x[(o * src_cols + start + c) * elem + e];
}

__global__ void ArgmaxKernel(const float *x, int64_t *out, size_t rows,
                             size_t cols) {
  const size_t r = blockIdx.x;
  if (r >= rows) return;
  // block 内并行求 max 下标
  __shared__ float vals[kBlock];
  __shared__ int idxs[kBlock];
  const float *row = x + r * cols;
  float local = -3.4e38f;
  int local_i = 0;
  for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
    if (row[c] > local) {
      local = row[c];
      local_i = static_cast<int>(c);
    }
  }
  vals[threadIdx.x] = local;
  idxs[threadIdx.x] = local_i;
  __syncthreads();
  if (threadIdx.x == 0) {
    float best = vals[0];
    int bi = 0;
    for (int i = 1; i < blockDim.x; ++i) {
      if (vals[i] > best) {
        best = vals[i];
        bi = i;
      }
    }
    out[r] = idxs[bi];
  }
}

}  // namespace

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
  if (ids.dtype() != core::DataType::INT64)
    return core::Status::UNSUPPORTED_TYPE;
  const size_t n = ids.num_elements();
  EmbeddingKernel<<<static_cast<unsigned>((n * hidden + 255) / 256), kBlock, 0,
                    Stream()>>>(
      ids.data<int64_t>(), table.data<float>(), result.data<float>(), n,
      hidden);
  return Finish();
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
  // 原地拷贝输入到 result 再旋转 (支持非原地语义)
  if (result.data<float>() != x.data<float>()) {
    if (cudaMemcpyAsync(result.data(), x.data(), x.num_bytes(),
                        cudaMemcpyDeviceToDevice, Stream()) != cudaSuccess)
      return core::Status::DEVICE_ERROR;
  }
  dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(heads));
  RopeKernel<<<grid, kBlock, 0, Stream()>>>(
      result.data<float>(), pos_ids.data<int64_t>(), rows, heads, dim, theta);
  return Finish();
}

core::Status ScaledDotProductAttention(const core::Context &ctx,
                                       const core::Tensor &q,
                                       const core::Tensor &k,
                                       const core::Tensor &v, bool is_causal,
                                       core::Tensor &result) {
  (void)ctx;
  if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3 ||
      q.dtype() != core::DataType::FLOAT32)
    return core::Status::UNSUPPORTED_TYPE;
  const size_t sq = q.shape()[0], hq = q.shape()[1], d = q.shape()[2];
  const size_t skv = k.shape()[0], hkv = k.shape()[1];
  if (k.shape()[2] != d || v.shape()[0] != skv || v.shape()[1] != hkv ||
      v.shape()[2] != d || hq % hkv != 0)
    return core::Status::SHAPE_MISMATCH;
  core::Status st = result.Resize({sq, hq, d});
  if (st != core::Status::OK) return st;
  dim3 grid(static_cast<unsigned>(sq), static_cast<unsigned>(hq));
  SdpaKernel<<<grid, 1, skv * sizeof(float), Stream()>>>(
      q.data<float>(), k.data<float>(), v.data<float>(),
      result.data<float>(), sq, hq, skv, hkv, d, is_causal ? 1 : 0);
  return Finish();
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
  const size_t elem = core::GetDataTypeSize(a.dtype());
  if (ax == nd - 1) {
    const size_t a_cols = a.shape().back();
    const size_t b_cols = b.shape().back();
    const size_t outer = a.num_elements() / a_cols;
    ConcatLastDimKernel<<<static_cast<unsigned>(
                              (outer * (a_cols + b_cols) + 255) / 256),
                          kBlock, 0, Stream()>>>(
        static_cast<const char *>(a.data()), static_cast<const char *>(b.data()),
        static_cast<char *>(result.data()), outer, a_cols, b_cols, elem);
  } else {
    // 非 last 维: outer 块整体拼接, 用两次 memcpyAsync
    size_t outer = 1;
    for (size_t i = 0; i < ax; ++i) outer *= a.shape()[i];
    const size_t a_seg = a.num_elements() / outer * elem;
    const size_t b_seg = b.num_elements() / outer * elem;
    char *rp = static_cast<char *>(result.data());
    const char *ap = static_cast<const char *>(a.data());
    const char *bp = static_cast<const char *>(b.data());
    for (size_t o = 0; o < outer; ++o) {
      if (cudaMemcpyAsync(rp, ap + o * a_seg, a_seg, cudaMemcpyDeviceToDevice,
                          Stream()) != cudaSuccess)
        return core::Status::DEVICE_ERROR;
      rp += a_seg;
      if (cudaMemcpyAsync(rp, bp + o * b_seg, b_seg, cudaMemcpyDeviceToDevice,
                          Stream()) != cudaSuccess)
        return core::Status::DEVICE_ERROR;
      rp += b_seg;
    }
  }
  return Finish();
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
  const size_t elem = core::GetDataTypeSize(x.dtype());
  if (axis == x.ndim() - 1) {
    const size_t src_cols = x.shape().back();
    const size_t outer = x.num_elements() / src_cols;
    SliceLastDimKernel<<<static_cast<unsigned>((outer * length + 255) / 256),
                         kBlock, 0, Stream()>>>(
        static_cast<const char *>(x.data()), static_cast<char *>(result.data()),
        outer, src_cols, start, length, elem);
  } else {
    size_t outer = 1;
    for (size_t i = 0; i < axis; ++i) outer *= x.shape()[i];
    const size_t src_len = x.shape()[axis];
    const size_t row = x.num_elements() / x.shape()[axis] / outer;
    const size_t seg = row * elem;
    const char *xp = static_cast<const char *>(x.data());
    char *rp = static_cast<char *>(result.data());
    for (size_t o = 0; o < outer; ++o) {
      if (cudaMemcpyAsync(rp, xp + ((o * src_len) + start) * seg,
                          length * seg, cudaMemcpyDeviceToDevice, Stream()) !=
          cudaSuccess)
        return core::Status::DEVICE_ERROR;
      rp += length * seg;
    }
  }
  return Finish();
}

core::Status ArgmaxLastDim(const core::Context &ctx, const core::Tensor &x,
                           core::Tensor &result) {
  (void)ctx;
  if (x.ndim() < 1 || x.dtype() != core::DataType::FLOAT32)
    return core::Status::UNSUPPORTED_TYPE;
  const size_t cols = x.shape().back();
  const size_t rows = x.num_elements() / cols;
  core::Status st = result.Resize({rows});
  if (st != core::Status::OK) return st;
  if (result.dtype() != core::DataType::INT64)
    return core::Status::UNSUPPORTED_TYPE;
  ArgmaxKernel<<<static_cast<unsigned>(rows), kBlock, 0, Stream()>>>(
      x.data<float>(), result.data<int64_t>(), rows, cols);
  return Finish();
}

}  // namespace cuda
}  // namespace ops
}  // namespace axono
