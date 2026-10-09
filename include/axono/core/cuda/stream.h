// Axono v0.2 — CUDA 当前流管理
//
// 所有 CUDA kernel 提交统一走 AxonoCurrentStream():
//   - 默认: 0 (legacy default stream), 行为与旧版完全一致;
//   - Graph 捕获期间: 捕获流 (由绑定层设置), 使 kernel 被记录进图;
//   - 未来多流/事件扩展预留。
//
// 同步策略: 算子内的 cudaDeviceSynchronize 在捕获期间被 IsCapturing()
// 抑制 (见 capture.h), 因此捕获体内只做异步提交, 回放后统一同步。
#pragma once

#include <cuda_runtime.h>

#include "axono/core/macros.h"

namespace axono {
namespace core {
namespace cuda {

// 当前线程应使用的提交流 (捕获时为捕获流, 否则默认流 0)
AXONO_EXPORT cudaStream_t AxonoCurrentStream();

// 内部: 设置/恢复当前线程捕获流 (绑定层经 BeginGraphCapture 使用)
AXONO_EXPORT void SetCaptureStream(cudaStream_t stream);

}  // namespace cuda
}  // namespace core
}  // namespace axono
