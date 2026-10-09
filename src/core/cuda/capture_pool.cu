// Axono v0.2 — 图捕获缓存池实现
#include <cuda_runtime.h>

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "axono/core/cuda/capture_pool.h"

namespace axono {
namespace core {
namespace cuda {

namespace {
std::mutex g_pool_mutex;
// 尺寸 -> 可复用块队列 (FIFO; warmup 与正式捕获的分配顺序一致)
std::unordered_map<size_t, std::vector<void*>> g_pool;
}  // namespace

void* CapturePoolAcquire(size_t bytes) {
  std::lock_guard<std::mutex> lk(g_pool_mutex);
  auto it = g_pool.find(bytes);
  if (it == g_pool.end() || it->second.empty()) return nullptr;
  void* ptr = it->second.back();
  it->second.pop_back();
  return ptr;
}

void CapturePoolRegister(size_t bytes, void* ptr) {
  std::lock_guard<std::mutex> lk(g_pool_mutex);
  g_pool[bytes].push_back(ptr);
}

namespace {
thread_local bool t_pool_recording = false;
}  // namespace

void SetCapturePoolRecording(bool enable) { t_pool_recording = enable; }
bool CapturePoolRecording() { return t_pool_recording; }

void ResetCapturePool() {
  std::unordered_map<size_t, std::vector<void*>> drained;
  {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    drained.swap(g_pool);
  }
  for (auto& kv : drained) {
    for (void* ptr : kv.second) cudaFree(ptr);
  }
}

}  // namespace cuda
}  // namespace core
}  // namespace axono
