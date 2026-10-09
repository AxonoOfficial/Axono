// Axono v0.2 — CUDA Graph 捕获与回放
//
// 用法 (C++ 绑定层): BeginGraphCapture -> (在捕获流上提交 kernels,
// 全局 IsCapturing() 为 true 时算子跳过同步) -> EndGraphCapture ->
// CudaGraphExec::Replay (任意次)。
//
// 注意: 捕获期间涉及的所有张量内存地址必须不变 (Axono 的 Tensor
// 分配后地址固定, 满足该假设); 捕获体内不得有 H2D/D2H 拷贝。
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "axono/core/macros.h"
#include "axono/core/types.h"

namespace axono {
namespace core {
namespace cuda {

// ---------------------------------------------------------------------------
// CudaGraphExec — 一个已实例化的 CUDA Graph (move-only)
// ---------------------------------------------------------------------------
class AXONO_EXPORT CudaGraphExec {
 public:
  CudaGraphExec() = default;
  ~CudaGraphExec();

  CudaGraphExec(const CudaGraphExec &) = delete;
  CudaGraphExec &operator=(const CudaGraphExec &) = delete;

  // 结束捕获并实例化 (capture_stream 处于 BeginCapture 之后)
  Status Finalize(cudaStream_t capture_stream);

  // 回放 (可多次调用, 异步)
  Status Replay(cudaStream_t stream);

  // 设备级同步 (回放后取结果前调用)
  Status Sync();

  bool IsCaptured() const;
  size_t NumNodes() const;

  // 销毁并回到未捕获状态 (可重新 capture)
  void Reset();

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  cudaStream_t capture_stream_ = nullptr;  // replay 必须回到此流
  size_t num_nodes_ = 0;
};

// ---------------------------------------------------------------------------
// 捕获流程辅助 (绑定层用)
// ---------------------------------------------------------------------------

// 创建捕获流并开始捕获; 成功时 *out_stream 为捕获流。
AXONO_EXPORT Status BeginGraphCapture(cudaStream_t *out_stream);

// 结束捕获并实例化到 exec。捕获流不销毁: 图内 stream-ordered 分配
// 节点与内存池绑定在该流上, replay 也要回到此流。
AXONO_EXPORT Status EndGraphCapture(cudaStream_t stream, CudaGraphExec *exec);

// 异常路径: 取消捕获并销毁流, exec 保持未捕获状态。
AXONO_EXPORT Status AbortGraphCapture(cudaStream_t stream);

}  // namespace cuda
}  // namespace core
}  // namespace axono
