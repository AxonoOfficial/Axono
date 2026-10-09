// Axono v0.2 — 图捕获缓存池
//
// 背景: 部分 CUDA 版本 (含 V100 + CUDA 12.8 实测) 上, 含 alloc 节点的
// CUDA Graph 同一 exec 第二次 cudaGraphLaunch 报 invalid argument。
// 因此捕获期间不得入图任何 alloc/free 节点 —— 改为预分配:
//
//   1. Python 层 capture() 在 BeginCapture 之前先真实跑一遍 fn
//      (warmup), 捕获池记下每个尺寸的块地址并持引用;
//   2. 正式捕获时, 同尺寸分配直接命中缓存 (地址不变), 图内只有
//      kernel 节点 —— 结果正确的前提是 kernel 完全覆盖输出, 实际
//      成立 (matmul/add/relu 均全量写);
//   3. 图 reset 时整池释放。
//
// 池是全局单映射 (size -> ptr), 因为同一捕获流上分配是顺序的。
#pragma once

#include <cstddef>

#include "axono/core/macros.h"

namespace axono {
namespace core {
namespace cuda {

// 查缓存: 命中返回地址并从缓存摘出 (一个块只给一次分配), 未命中 nullptr
AXONO_EXPORT void* CapturePoolAcquire(size_t bytes);

// 登记: warmup (登记模式) 期间的真实分配进入缓存 (池持引用直到 Reset)
AXONO_EXPORT void CapturePoolRegister(size_t bytes, void* ptr);

// 登记模式开关: 开启后 CudaAllocateStorage 的普通分配在返回前登记进池。
// Python 层 warmup 时开启, 让捕获复用这些地址。
AXONO_EXPORT void SetCapturePoolRecording(bool enable);
AXONO_EXPORT bool CapturePoolRecording();

// 清空并释放整个池 (图 reset 时调用)
AXONO_EXPORT void ResetCapturePool();

}  // namespace cuda
}  // namespace core
}  // namespace axono
