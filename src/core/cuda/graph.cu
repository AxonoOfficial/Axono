// Axono v0.2 — CUDA 当前流管理实现 + graph.cu 合并
#include <cuda_runtime.h>

#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "axono/core/cuda/capture.h"
#include "axono/core/cuda/graph.h"
#include "axono/core/cuda/stream.h"

namespace axono {
namespace core {
namespace cuda {

namespace {
thread_local bool t_capturing = false;
thread_local cudaStream_t t_capture_stream = nullptr;

// 捕获期间挂起的待释放指针 (分配流, 指针); 捕获结束后统一释放。
// 全局 (非线程局部): 图输出张量可能在别的线程析构。
std::mutex g_deferred_mutex;
std::vector<std::pair<cudaStream_t, void*>> g_deferred_frees;
}  // namespace

bool IsCapturing() { return t_capturing; }
void SetCapturing(bool v) { t_capturing = v; }

bool DeferredFreeAsync(void* ptr, cudaStream_t alloc_stream) {
  if (t_capturing) {
    std::lock_guard<std::mutex> lk(g_deferred_mutex);
    g_deferred_frees.emplace_back(alloc_stream, ptr);
    return true;
  }
  cudaFreeAsync(ptr, alloc_stream);
  return false;
}

void FlushDeferredFrees() {
  std::vector<std::pair<cudaStream_t, void*>> pending;
  {
    std::lock_guard<std::mutex> lk(g_deferred_mutex);
    pending.swap(g_deferred_frees);
  }
  for (auto& p : pending) cudaFreeAsync(p.second, p.first);
}

cudaStream_t AxonoCurrentStream() {
  return t_capturing ? t_capture_stream : nullptr;
}

void SetCaptureStream(cudaStream_t stream) { t_capture_stream = stream; }

// ---------------------------------------------------------------------------
// CudaGraphExec
// ---------------------------------------------------------------------------

#define AXONO_CUDA_CHECK(expr)                                         \
  do {                                                                 \
    cudaError_t e_ = (expr);                                           \
    if (e_ != cudaSuccess) {                                           \
      throw std::runtime_error(std::string("CUDA error: ") +           \
                               cudaGetErrorString(e_) + " at " #expr); \
    }                                                                  \
  } while (0)

CudaGraphExec::~CudaGraphExec() { Reset(); }

Status CudaGraphExec::Finalize(cudaStream_t capture_stream) {
  if (graph_ != nullptr || exec_ != nullptr) {
    return Status::INVALID_ARGUMENT;  // 已捕获, 需先 Reset
  }
  try {
    AXONO_CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph_));
    if (graph_ == nullptr) return Status::INTERNAL_ERROR;

    size_t num_nodes = 0;
    AXONO_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &num_nodes));
    num_nodes_ = num_nodes;

    // 空图视为错误: 说明捕获体内没有提交任何 CUDA 算子
    // (常见原因: 张量在 CPU 上, 或整段都是 host 侧代码)
    if (num_nodes == 0) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
      return Status::INVALID_ARGUMENT;
    }

    cudaGraphExec_t exec = nullptr;
    // AutoFreeOnLaunch: 每次 launch 自动重发图内的 free 节点。实测
    // (V100/CUDA 12.8) 含 alloc 节点的图不加此 flag 时第二次
    // cudaGraphLaunch 报 invalid argument, 加此 flag 后可正常重复
    // 回放 —— 是原生 stream-ordered 分配入图路径的必要开关。
    cudaError_t err = cudaGraphInstantiateWithFlags(
        &exec, graph_, cudaGraphInstantiateFlagAutoFreeOnLaunch);
    if (err != cudaSuccess) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
      return Status::INTERNAL_ERROR;
    }
    exec_ = exec;
    capture_stream_ = capture_stream;  // 保存: replay 必须回到此流
    return Status::OK;
  } catch (const std::exception &) {
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
    }
    return Status::INTERNAL_ERROR;
  }
}

Status CudaGraphExec::Replay(cudaStream_t) {
  if (exec_ == nullptr) return Status::INVALID_ARGUMENT;
  cudaGetLastError();  // 清除残留错误
  cudaError_t e1 = cudaGraphLaunch(exec_, capture_stream_);
  if (e1 != cudaSuccess) {
    std::fprintf(stderr, "[axono] cudaGraphLaunch failed: %s\n",
                 cudaGetErrorString(e1));
    return Status::INTERNAL_ERROR;
  }
  cudaError_t e2 = cudaStreamSynchronize(capture_stream_);
  if (e2 != cudaSuccess) {
    std::fprintf(stderr, "[axono] graph replay sync failed: %s\n",
                 cudaGetErrorString(e2));
    return Status::INTERNAL_ERROR;
  }
  return Status::OK;
}

Status CudaGraphExec::Sync() {
  return cudaDeviceSynchronize() == cudaSuccess ? Status::OK
                                                : Status::INTERNAL_ERROR;
}

bool CudaGraphExec::IsCaptured() const { return exec_ != nullptr; }

size_t CudaGraphExec::NumNodes() const { return num_nodes_; }

void CudaGraphExec::Reset() {
  if (exec_ != nullptr) {
    // Replay 内部已同步, 这里再兜底一次
    if (capture_stream_ != nullptr) cudaStreamSynchronize(capture_stream_);
    cudaGraphExecDestroy(exec_);
    exec_ = nullptr;
  }
  // 注意: 捕获流不销毁。图捕获期间分配的 Tensor 来自该流的内存池,
  // 这些张量可能在图 reset 甚至对象析构之后才在 Python 侧释放 —— 届时
  // cudaFreeAsync 仍需一个有效流。每个图泄漏一条流 (进程级, 数量极少),
  // 由驱动在进程退出时回收, 换取内存安全。
  capture_stream_ = nullptr;
  if (graph_ != nullptr) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  num_nodes_ = 0;
}

// ---------------------------------------------------------------------------
// 捕获流程辅助
// ---------------------------------------------------------------------------

Status BeginGraphCapture(cudaStream_t *out_stream) {
  cudaStream_t stream = nullptr;
  // 预热失败残留的 sticky error 会让后续一切 CUDA 调用失败 —— 清掉。
  cudaGetLastError();
  try {
    AXONO_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    AXONO_CUDA_CHECK(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    t_capture_stream = stream;
    t_capturing = true;
    *out_stream = stream;
    return Status::OK;
  } catch (const std::exception &) {
    if (stream) cudaStreamDestroy(stream);
    t_capture_stream = nullptr;
    t_capturing = false;
    return Status::INTERNAL_ERROR;
  }
}

Status EndGraphCapture(cudaStream_t stream, CudaGraphExec *exec) {
  t_capturing = false;
  t_capture_stream = nullptr;
  // 注意: 不销毁捕获流 —— 图的 stream-ordered 分配节点与其内存池
  // 都绑定在该流上, 流一旦销毁, 图输出张量在 Python 侧析构时对死池
  // 释放会崩溃。流留到进程结束 (或下次 capture 覆盖) 由驱动回收。
  Status st = exec->Finalize(stream);
  // 捕获结束: 释放捕获期间挂起的临时张量 (它们只用于构建图, 图的
  // 内存由 mem pool 持有, 延迟释放不会影响已入图的节点)
  FlushDeferredFrees();
  return st;
}

Status AbortGraphCapture(cudaStream_t stream) {
  t_capturing = false;
  t_capture_stream = nullptr;
  cudaGraph_t aborted = nullptr;
  cudaStreamEndCapture(stream, &aborted);
  if (aborted) cudaGraphDestroy(aborted);
  cudaStreamDestroy(stream);  // 未成图, 无 pool 引用, 可安全销毁
  return Status::OK;
}

}  // namespace cuda
}  // namespace core
}  // namespace axono
