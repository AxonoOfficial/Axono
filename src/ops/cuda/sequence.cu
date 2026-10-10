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
  // 异步模式下仅检查 launch 错误 (捕获中同样跳过); 同步交给 D2H 读回。
  return core::cuda::MaybeSync() == cudaSuccess ? core::Status::OK
                                               : core::Status::INTERNAL_ERROR;
}

// ---- block 内 max / sum 归约 (warp shuffle) ----
template <typename T>
__device__ inline T BlockReduceMax(T v) {
  for (int off = 16; off > 0; off >>= 1) {
    const T o = __shfl_down_sync(0xffffffff, v, off);
    if (o > v) v = o;
  }
  __shared__ T warp[32];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) warp[wid] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  if (threadIdx.x < 32) {
    v = threadIdx.x < nw ? warp[threadIdx.x] : T(-3.4e38);
    for (int off = 16; off > 0; off >>= 1) {
      const T o = __shfl_down_sync(0xffffffff, v, off);
      if (o > v) v = o;
    }
    if (threadIdx.x == 0) warp[0] = v;
  }
  __syncthreads();
  return warp[0];
}

template <typename T>
__device__ inline T BlockReduceSum(T v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  __shared__ T warp[32];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) warp[wid] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  if (threadIdx.x < 32) {
    v = threadIdx.x < nw ? warp[threadIdx.x] : T(0);
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffff, v, off);
    if (threadIdx.x == 0) warp[0] = v;
  }
  __syncthreads();
  return warp[0];
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

// SDPA: 每 (t, head) 一个线程块, block 内并行: 分数 (warp 分片点积) +
// block 归约 max/sum + 加权 V。d == 4 的倍数时用 float4 向量化访存。
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

  extern __shared__ float scores[];
  // 1) 分数: 每 warp 分片 kv, block 内并行点积
  for (size_t s = threadIdx.x; s < kv_end; s += blockDim.x) {
    const float *kr = k + (s * hkv + hk) * d;
    float acc = 0.0f;
    size_t i = 0;
    if (d % 4 == 0) {
      const float4 *q4 = reinterpret_cast<const float4 *>(qr);
      const float4 *k4 = reinterpret_cast<const float4 *>(kr);
      for (; i < d; i += 4) {
        const float4 a = q4[i / 4];
        const float4 b = k4[i / 4];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
      }
    } else {
      for (; i < d; ++i) acc += qr[i] * kr[i];
    }
    scores[s] = acc * scale;
  }
  __syncthreads();

  // 2) 行最大 (block 归约)
  float local_max = -3.4e38f;
  for (size_t s = threadIdx.x; s < kv_end; s += blockDim.x)
    local_max = scores[s] > local_max ? scores[s] : local_max;
  const float mx = BlockReduceMax<float>(local_max);

  // 3) exp + sum
  float local_sum = 0.0f;
  for (size_t s = threadIdx.x; s < kv_end; s += blockDim.x) {
    scores[s] = expf(scores[s] - mx);
    local_sum += scores[s];
  }
  const float sum = BlockReduceSum<float>(local_sum);
  const float inv = 1.0f / sum;

  // 4) 加权 V: 每 warp 分片输出维
  for (size_t i = threadIdx.x; i < d; i += blockDim.x) {
    float acc = 0.0f;
    for (size_t s = 0; s < kv_end; ++s)
      acc += scores[s] * v[(s * hkv + hk) * d + i];
    outr[i] = acc * inv;
  }
}

// RoPE 变体 1: (cos, sin) 直接旋转 — 每 (r,h) 一线程, rotate_half 配对
__global__ void RopeCosSinKernel(const float *x, const float *cos,
                                 const float *sin, float *out, size_t rows,
                                 size_t heads, size_t dim) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = rows * heads;
  if (idx >= total) return;
  const size_t r = idx / heads;
  const float *src = x + idx * dim;
  float *dst = out + idx * dim;
  const float *cr = cos + r * dim;
  const float *sr = sin + r * dim;
  const size_t half = dim / 2;
  for (size_t i = 0; i < half; ++i) {
    const float a = src[i], b = src[i + half];
    dst[i] = a * cr[i] - b * sr[i];
    dst[i + half] = b * cr[i + half] + a * sr[i + half];
  }
}

// RoPE 变体 2: 3D 位置 (T/H/W) 交错 stride-3 频率重组 — 每 (r,h) 一线程
__global__ void RopeThdKernel(const float *x, const int64_t *pos,
                              const float *inv_freq, float *out, size_t rows,
                              size_t heads, size_t dim, size_t h_lim,
                              size_t w_lim) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = rows * heads;
  if (idx >= total) return;
  const size_t r = idx / heads;
  const float *src = x + idx * dim;
  float *dst = out + idx * dim;
  const float pt = static_cast<float>(pos[r]);
  const float ph = static_cast<float>(pos[rows + r]);
  const float pw = static_cast<float>(pos[2 * rows + r]);
  const size_t half = dim / 2;
  for (size_t i = 0; i < half; ++i) {
    const float f = inv_freq[i];
    float angle = pt * f;
    if (i % 3 == 1 && i < h_lim)
      angle = ph * f;
    else if (i % 3 == 2 && i < w_lim)
      angle = pw * f;
    const float c = cosf(angle), s = sinf(angle);
    const float a = src[i], b = src[i + half];
    dst[i] = a * c - b * s;
    dst[i + half] = b * c + a * s;
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

// 通用 strided slice (非 last 维): 输出按 (outer, length, row) 展平, 每
// 元素定位源位置。替换原 "逐 outer cudaMemcpyAsync" (outer 大时 launch 开销
// 主导)。线程按元素并行, 内层按 elem 字节拷贝 (elem<=8)。
__global__ void SliceStridedKernel(const char *x, char *out, size_t total_elems,
                                   size_t length, size_t row, size_t src_len,
                                   size_t start, unsigned elem) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  const size_t r = idx % row;
  const size_t t = idx / row;
  const size_t li = t % length;
  const size_t o = t / length;
  const char *sp = x + ((o * src_len + start + li) * row + r) * elem;
  char *dp = out + idx * elem;
  for (unsigned e = 0; e < elem; ++e) dp[e] = sp[e];
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
  const size_t total = rows * heads;
  RopeCosSinKernel<<<static_cast<unsigned>((total + kBlock - 1) / kBlock),
                     kBlock, 0, Stream()>>>(
      x.data<float>(), cos.data<float>(), sin.data<float>(),
      result.data<float>(), rows, heads, dim);
  return Finish();
}

// M-RoPE 的 cos/sin 表: pos3 (3, seq) -> cos/sin (seq, dim), 交错 stride-3
// 频率重组 (T 基础, H/W 覆盖 idx%3==1/2 至各自 section*3), 与 HF 等价。
// 输出为 concat([ang, ang]) 的 cos/sin (即 dim 维, 前后两半相同)。
__global__ void MropeCosSinKernel(const int64_t *pos, const float *inv_freq,
                                  float *cos_out, float *sin_out,
                                  size_t rows, size_t half, size_t h_lim,
                                  size_t w_lim) {
  const size_t r = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float pt = static_cast<float>(pos[r]);
  const float ph = static_cast<float>(pos[rows + r]);
  const float pw = static_cast<float>(pos[2 * rows + r]);
  float *cr = cos_out + r * (2 * half);
  float *sr = sin_out + r * (2 * half);
  for (size_t i = 0; i < half; ++i) {
    const float f = inv_freq[i];
    float angle = pt * f;
    if (i % 3 == 1 && i < h_lim)
      angle = ph * f;
    else if (i % 3 == 2 && i < w_lim)
      angle = pw * f;
    const float c = cosf(angle);
    const float sn = sinf(angle);
    cr[i] = c;
    cr[i + half] = c;
    sr[i] = sn;
    sr[i + half] = sn;
  }
}

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
  const size_t h_lim = static_cast<size_t>(h_sec) * 3;
  const size_t w_lim = static_cast<size_t>(w_sec) * 3;
  MropeCosSinKernel<<<static_cast<unsigned>((rows + kBlock - 1) / kBlock),
                      kBlock, 0, Stream()>>>(
      pos.data<int64_t>(), inv_freq.data<float>(), cos_out.data<float>(),
      sin_out.data<float>(), rows, half, h_lim, w_lim);
  return Finish();
}

core::Status RopeThd(const core::Context &ctx, const core::Tensor &x,
                     const core::Tensor &pos, const core::Tensor &inv_freq,
                     int t_sec, int h_sec, int w_sec, core::Tensor &result) {
  (void)ctx;
  (void)t_sec;
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
  core::Status st = result.Resize(x.shape());
  if (st != core::Status::OK) return st;
  const size_t total = rows * heads;
  const size_t h_lim = static_cast<size_t>(h_sec) * 3;
  const size_t w_lim = static_cast<size_t>(w_sec) * 3;
  RopeThdKernel<<<static_cast<unsigned>((total + kBlock - 1) / kBlock),
                  kBlock, 0, Stream()>>>(
      x.data<float>(), pos.data<int64_t>(), inv_freq.data<float>(),
      result.data<float>(), rows, heads, dim, h_lim, w_lim);
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
  SdpaKernel<<<grid, kBlock, skv * sizeof(float), Stream()>>>(
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
    // 单 kernel 元素级 strided 拷贝 (替换逐 outer cudaMemcpyAsync — 每段一次
    // launch 在 outer 大时 (如 qkv 拆分 outer=256) 开销 ~1ms)
    const size_t total_elems = outer * length * row;
    const unsigned grid1 =
        static_cast<unsigned>((total_elems + 255) / 256);
    SliceStridedKernel<<<grid1, 256, 0, Stream()>>>(
        xp, rp, total_elems, length, row, src_len, start,
        static_cast<unsigned>(elem));
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
