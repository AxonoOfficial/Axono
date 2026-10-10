// GQA Attention (CUDA): decode (q_len=1) 专用 split-K 优化 kernel。
//
// 相比通用 SdpaKernel 的 decode 路径 (每 (t,head) 一个 block, 加权 V 串行
// 循环, skv 大时单 block 瓶颈), 此 kernel 采用 flash-decoding 标准 split-K:
//   phase 1: grid = (hkv, num_splits); 每 block 用 group 个 warp 各处理一个
//            q-head, kv 分片并行计算局部 (max, sum, 加权 V 部分);
//   phase 2: merge kernel 跨分片做 online-softmax 归一化合并。
// K/V 每个 kv-head 只从 global 读一次 (group 内 q-head 共享), 输出维并行。
#include "axono/ops/cuda/fused.h"

#include <cuda_runtime.h>
#include <math.h>

#include <vector>

#include "axono/core/cuda/capture.h"
#include "axono/ops/cuda/sequence.h"
#include "axono/core/cuda/stream.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cuda {
namespace {

constexpr int kMaxSplits = 256;

// phase 1: 每 block = (kv_head, split)。blockDim = group*32 线程,
// 每 warp 负责一个 q-head; warp 内 32 lane 对本分片的 kv 逐个算点积。
__global__ void GqaDecodeSplitKernel(const float *q, const float *k,
                                     const float *v, float *partial_out,
                                     float *partial_max, float *partial_sum,
                                     size_t hq, size_t hkv, size_t d,
                                     size_t skv, size_t n_splits) {
  const size_t hk = blockIdx.x;
  const size_t sp = blockIdx.y;
  const size_t group = hq / hkv;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (static_cast<size_t>(warp) >= group) return;
  const size_t h = hk * group + warp;

  // 本分片的 kv 范围
  const size_t chunk = (skv + n_splits - 1) / n_splits;
  const size_t s0 = sp * chunk;
  const size_t s1 = min(s0 + chunk, skv);

  const float *qr = q + h * d;
  const float scale = rsqrtf(static_cast<float>(d));

  // 局部 online-softmax: 累计 (m, l) 并累加加权 V (输出维由 lane 分片)
  // 设计: 每线程持有一个输出维累加器 → 输出维 = 32 lane 循环覆盖 d。
  // scores 需要两遍 (先 m/l, 再加权 V) — 或用在线重缩放。用在线重缩放:
  extern __shared__ float sv[];  // group * d: 每 warp 的输出累加器

  float m = -3.4e38f, l = 0.0f;
  float *acc = sv + warp * d;

  for (size_t i = lane; i < d; i += 32) acc[i] = 0.0f;
  __syncwarp();

  for (size_t s = s0; s < s1; ++s) {
    const float *kr = k + (s * hkv + hk) * d;
    // warp 并行点积
    float acc_qk = 0.0f;
    for (size_t i = lane; i < d; i += 32) acc_qk += qr[i] * kr[i];
    // warp 归约
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc_qk += __shfl_down_sync(0xffffffff, acc_qk, off);
    acc_qk = __shfl_sync(0xffffffff, acc_qk, 0) * scale;

    const float mn = fmaxf(m, acc_qk);
    const float corr = expf(m - mn);       // 旧权重校正
    const float p = expf(acc_qk - mn);     // 新 score 权重
    l = l * corr + p;
    m = mn;
    // 输出累加: acc = acc * corr + p * v[s]
    const float *vr = v + (s * hkv + hk) * d;
    for (size_t i = lane; i < d; i += 32) acc[i] = acc[i] * corr + p * vr[i];
  }
  __syncwarp();

  // 写 partial: partial_out (n_splits, hq, d), partial_max/sum (n_splits, hq)
  float *po = partial_out + (sp * hq + h) * d;
  for (size_t i = lane; i < d; i += 32) po[i] = acc[i];
  if (lane == 0) {
    partial_max[sp * hq + h] = m;
    partial_sum[sp * hq + h] = l;
  }
}

// phase 2: 跨 split 合并。grid = (hq), 每 warp 一头; lane 分片输出维。
__global__ void GqaDecodeMergeKernel(const float *partial_out,
                                     const float *partial_max,
                                     const float *partial_sum, float *out,
                                     size_t hq, size_t d, size_t n_splits,
                                     size_t n_valid_splits) {
  const size_t h = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (static_cast<size_t>(warp) != h % ((blockDim.x >> 5) == 0 ? 1 : (blockDim.x >> 5)))
    return;  // 每 block 至多处理几个头时才用; 此处 grid=hq, blockDim=32
  (void)warp;

  // 1) 全局 max
  float m = -3.4e38f;
  for (size_t sp = 0; sp < n_valid_splits; ++sp) {
    const float mv = partial_max[sp * hq + h];
    m = fmaxf(m, mv);
  }
  // 2) 归一化 sum
  float l = 0.0f;
  for (size_t sp = 0; sp < n_valid_splits; ++sp)
    l += partial_sum[sp * hq + h] * expf(partial_max[sp * hq + h] - m);
  const float inv = 1.0f / l;

  // 3) 加权合并
  float *outr = out + h * d;
  for (size_t i = lane; i < d; i += 32) {
    float acc = 0.0f;
    for (size_t sp = 0; sp < n_valid_splits; ++sp) {
      const float w = expf(partial_max[sp * hq + h] - m) * inv;
      acc += w * partial_out[(sp * hq + h) * d + i];
    }
    outr[i] = acc;
  }
}

inline cudaStream_t Stream() { return core::cuda::AxonoCurrentStream(); }
core::Status Finish() {
  return core::cuda::MaybeSync() == cudaSuccess ? core::Status::OK
                                                : core::Status::INTERNAL_ERROR;
}

}  // namespace

core::Status GqaDecodeAttention(const core::Context &ctx, const core::Tensor &q,
                                const core::Tensor &k, const core::Tensor &v,
                                core::Tensor &result) {
  (void)ctx;
  if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3 ||
      q.dtype() != core::DataType::FLOAT32)
    return core::Status::UNSUPPORTED_TYPE;
  const size_t sq = q.shape()[0], hq = q.shape()[1], d = q.shape()[2];
  const size_t skv = k.shape()[0], hkv = k.shape()[1];
  if (sq != 1) return core::Status::INVALID_ARGUMENT;  // decode 专用
  // 短上下文 split-K 收益不抵 launch/合并开销, 直接走通用 SDPA。
  if (skv < 2048)
    return ScaledDotProductAttention(ctx, q, k, v, false, result);
  if (k.shape()[2] != d || v.shape()[0] != skv || v.shape()[1] != hkv ||
      v.shape()[2] != d || hq % hkv != 0)
    return core::Status::SHAPE_MISMATCH;
  const size_t group = hq / hkv;

  // split 数: 每 split 至少 256 kv, 至多 kMaxSplits, 至少 1
  size_t n_splits = (skv + 63) / 64;
  if (n_splits < 1) n_splits = 1;
  if (n_splits > kMaxSplits) n_splits = kMaxSplits;

  core::Status st = result.Resize({1, hq, d});
  if (st != core::Status::OK) return st;

  // 工作区: partial_out (n_splits*hq*d), max/sum (n_splits*hq) — 走池
  core::Tensor w_out(q.dtype(), {n_splits * hq * d}, q.device());
  core::Tensor w_max(q.dtype(), {n_splits * hq}, q.device());
  core::Tensor w_sum(q.dtype(), {n_splits * hq}, q.device());

  dim3 grid(static_cast<unsigned>(hkv), static_cast<unsigned>(n_splits));
  const size_t shmem = group * d * sizeof(float);
  if (shmem > 48 * 1024) return core::Status::INVALID_ARGUMENT;  // group*d 过大
  GqaDecodeSplitKernel<<<grid, static_cast<unsigned>(group * 32), shmem,
                         Stream()>>>(
      q.data<float>(), k.data<float>(), v.data<float>(),
      w_out.data<float>(), w_max.data<float>(), w_sum.data<float>(),
      hq, hkv, d, skv, n_splits);
  GqaDecodeMergeKernel<<<static_cast<unsigned>(hq), 32, 0, Stream()>>>(
      w_out.data<float>(), w_max.data<float>(), w_sum.data<float>(),
      result.data<float>(), hq, d, n_splits, n_splits);
  return Finish();
}

}  // namespace cuda
}  // namespace ops
}  // namespace axono
