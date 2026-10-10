// Axono v0.2 — CUDA 捕获状态 (全局, 线程局部)
//
// 捕获期间 (cudaStreamBeginCapture .. EndCapture) 禁止:
//   - cudaDeviceSynchronize / cudaStreamSynchronize (会使捕获失败)
//   - cudaMalloc / cudaFree (需用 cudaMallocAsync 或预分配)
// CUDA 算子内原本每次调用后都有 cudaDeviceSynchronize (保证正确性),
// 捕获时由本标志抑制, 回放后统一同步。
#pragma once

#include <cuda_runtime.h>

#include "axono/core/macros.h"

namespace axono {
namespace core {
namespace cuda {

// 当前线程是否处于 Graph 捕获中
AXONO_EXPORT bool IsCapturing();
AXONO_EXPORT void SetCapturing(bool capturing);

// 异步执行模式 (默认开): 非捕获路径的算子也不再 cudaDeviceSynchronize,
// 正确性由 default-stream 顺序 + D2H cudaMemcpy 隐式同步保证 (torch 同款
// 语义)。设为 false 可恢复旧行为 (每算子全设备同步, 便于调试)。
AXONO_EXPORT void SetAsyncMode(bool enable);
AXONO_EXPORT bool AsyncMode();

// 统一同步点: 仅在既非捕获又未开异步模式时真正 deviceSync。
// 返回 cudaSuccess 或错误码, 供算子入口返回 Status。
inline cudaError_t MaybeSync() {
  if (!IsCapturing() && !AsyncMode()) return cudaDeviceSynchronize();
  return cudaSuccess;
}

// 捕获期间的指针释放策略:
//   - 捕获中: 挂到延迟队列 (不入图 free 节点) —— 否则同一图 exec
//     的第二次 cudaGraphLaunch 会因 free/alloc 节点混排报
//     invalid argument (CUDA 12 限制; PyTorch 同样延迟释放);
//   - 非捕获中: 立即 cudaFreeAsync (若是捕获期分配的池指针, 仍发到
//     其分配流)。
// 返回 true 表示已延迟挂起, 由 EndGraphCapture 统一释放。
AXONO_EXPORT bool DeferredFreeAsync(void* ptr, cudaStream_t alloc_stream);

// 内部: 释放所有捕获期间挂起的指针 (EndGraphCapture 调用)。
AXONO_EXPORT void FlushDeferredFrees();

// RAII: 进入作用域置位, 退出恢复原值
class CaptureGuard {
 public:
  explicit CaptureGuard(bool value = true) : prev_(IsCapturing()) {
    SetCapturing(value);
  }
  ~CaptureGuard() { SetCapturing(prev_); }
  CaptureGuard(const CaptureGuard &) = delete;
  CaptureGuard &operator=(const CaptureGuard &) = delete;

 private:
  bool prev_;
};

}  // namespace cuda
}  // namespace core
}  // namespace axono
