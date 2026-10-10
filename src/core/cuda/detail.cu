// Axono/src/core/cuda/detail.cu
#include <cuda_runtime.h>
#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "axono/core/cuda/capture.h"
#include "axono/core/cuda/capture_pool.h"
#include "axono/core/cuda/stream.h"
#include "axono/core/tensor.h"

namespace axono {
namespace core {
namespace cuda {
namespace detail {

// ---------------------------------------------------------------------------
// 通用设备内存缓存池 (caching allocator)
// ---------------------------------------------------------------------------
// 动机: 每个算子输出都会新建 Tensor → 每次 cudaMalloc/cudaFree (~100-500us,
// 且串行化 host) 是 Qwen3-VL 前向实测的主要瓶颈之一 (每前向数百次分配)。
//
// 语义: 按字节数分桶缓存空闲块, 释放即归还 (shared_ptr deleter)。命中的块
// 仍执行 cudaMemset 归零 —— 与 cudaMalloc 路径保持一致语义 (Axono 保证新
// 张量初值为 0), 代价仅一次 D2D memset (~10us/4MB)。
//
// 线程安全: 全局 mutex (分配本身不频繁到需要分片)。
// 与图捕获的关系: 捕获路径完全绕过本池 (走 capture_pool 预分配), 互不影响。
namespace {
std::mutex g_free_mutex;
std::unordered_map<size_t, std::vector<void*>>& FreeMap() {
  static std::unordered_map<size_t, std::vector<void*>>* m =
      new std::unordered_map<size_t, std::vector<void*>>();
  return *m;
}

void* CacheAcquire(size_t bytes) {
  std::lock_guard<std::mutex> lk(g_free_mutex);
  auto& m = FreeMap();
  auto it = m.find(bytes);
  if (it == m.end() || it->second.empty()) return nullptr;
  void* ptr = it->second.back();
  it->second.pop_back();
  return ptr;
}
}  // namespace
}  // namespace detail

// 释放时归还缓存池 (供 detail.cu 内的 deleter 调用)
void ReturnToCache(size_t bytes, void* ptr) {
  std::lock_guard<std::mutex> lk(detail::g_free_mutex);
  detail::FreeMap()[bytes].push_back(ptr);
}

namespace detail {

// CUDA设备内存分配实现
std::shared_ptr<void> CudaAllocateStorage(size_t bytes, const std::string& device) {
    // 解析设备字符串
    int device_id = 0;
    if (device.size() > 5) {
        try {
            device_id = std::stoi(device.substr(5));
        } catch (...) {
            throw std::invalid_argument("Invalid CUDA device format: " + device);
        }
    }

    // 设备检测
    int cuda_device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&cuda_device_count);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA device detection failed: " + 
                               std::string(cudaGetErrorString(err)));
    }
    if (device_id < 0 || device_id >= cuda_device_count) {
        throw std::out_of_range("CUDA device " + std::to_string(device_id) + 
                              " out of range (0-" + std::to_string(cuda_device_count-1) + ")");
    }

    // 设置当前设备
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device: " + 
                               std::string(cudaGetErrorString(err)));
    }

    // 分配CUDA内存
    // 图捕获: 优先命中捕获缓存池 (绑定层 capture() 已在 warmup 阶段把
    // 同尺寸块缓存于此)。命中则图内零 alloc 节点 —— 实测部分 CUDA 版本
    // 上含 alloc 节点的图无法重复 launch (invalid argument), 预分配是
    // 唯一稳妥方案。
    // 非捕获: 普通 cudaMalloc; 登记模式 (warmup) 下额外登记进池。
    void* dev_ptr = nullptr;
    const bool capturing = axono::core::cuda::IsCapturing();
    cudaStream_t alloc_stream =
        capturing ? axono::core::cuda::AxonoCurrentStream() : nullptr;
    if (capturing) {
        dev_ptr = axono::core::cuda::CapturePoolAcquire(bytes);
        if (dev_ptr != nullptr) {
            // 命中缓存: 直接复用 (无需清零 —— 回放后 kernel 覆盖全部元素)
            return std::shared_ptr<void>(dev_ptr,
                                         [](void*) { /* 池持有, 不释放 */ });
        }
        // 池未命中兜底: cudaMallocAsync 会成为图内 alloc 节点, 首次
        // 回放可用, 但重复回放在部分 CUDA 版本上受限 —— 仅在 warmup
        // 覆盖不全 (用户捕获体依赖运行时分支) 时才会走到这里。
        err = cudaMallocAsync(&dev_ptr, bytes, alloc_stream);
        if (err != cudaSuccess) throw std::bad_alloc();
        err = cudaMemsetAsync(dev_ptr, 0, bytes, alloc_stream);
        if (err != cudaSuccess) {
            cudaFreeAsync(dev_ptr, alloc_stream);
            throw std::runtime_error("CUDA memset failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        return std::shared_ptr<void>(
            dev_ptr, [alloc_stream](void* ptr) {
              cudaFreeAsync(ptr, alloc_stream);
            });
    }
// 非捕获: 缓存池命中直接复用, 未命中 cudaMalloc; 释放归还缓存池。
    dev_ptr = CacheAcquire(bytes);
    if (dev_ptr != nullptr) {
      // 命中: 免 cudaMalloc/cudaFree; 语义仍保证零初始化 (见上注释)。
      // 异步入流 (与后续 kernel 同序, host 不空等)。
      err = cudaMemsetAsync(dev_ptr, 0, bytes,
                            axono::core::cuda::AxonoCurrentStream());
      if (err != cudaSuccess) {
        ReturnToCache(bytes, dev_ptr);
        throw std::runtime_error("CUDA memset failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
      return std::shared_ptr<void>(
          dev_ptr, [bytes](void* p) { ReturnToCache(bytes, p); });
    }
    err = cudaMalloc(&dev_ptr, bytes);
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }

    // 初始化为0 (捕获中异步, 同样入图)
    if (capturing) {
      err = cudaMemsetAsync(dev_ptr, 0, bytes, alloc_stream);
    } else {
      err = cudaMemset(dev_ptr, 0, bytes);
    }
    if (err != cudaSuccess) {
      if (capturing) {
        cudaFreeAsync(dev_ptr, alloc_stream);
      } else {
        cudaFree(dev_ptr); // 清理已分配的内存
      }
      throw std::runtime_error("CUDA memset failed: " +
                               std::string(cudaGetErrorString(err)));
    }

    // 返回带CUDA释放器的智能指针 (按分配方式选择释放路径)
    if (axono::core::cuda::CapturePoolRecording()) {
      // 登记模式 (warmup): 池持引用, 分配的块供捕获复用
      axono::core::cuda::CapturePoolRegister(bytes, dev_ptr);
      return std::shared_ptr<void>(dev_ptr,
                                   [](void*) { /* 池持有, 不释放 */ });
    }
    if (capturing) {
      return std::shared_ptr<void>(
          dev_ptr, [alloc_stream](void* ptr) {
            // 捕获中: 延迟释放, 不入图 free 节点
            axono::core::cuda::DeferredFreeAsync(ptr, alloc_stream);
          });
    }
    // 非捕获: 块释放时归还缓存池 (下次同尺寸分配直接命中, 免 cudaMalloc)
    return std::shared_ptr<void>(
        dev_ptr, [bytes](void* ptr) { axono::core::cuda::ReturnToCache(bytes, ptr); });
}

// 清空缓存池 (显存紧张 / 测试隔离时调用; 正常运行无需调用)
void ClearDeviceCache() {
  std::unordered_map<size_t, std::vector<void*>> drained;
  {
    std::lock_guard<std::mutex> lk(g_free_mutex);
    drained.swap(FreeMap());
  }
  for (auto& kv : drained) {
    for (void* ptr : kv.second) cudaFree(ptr);
  }
}

/**
 * @brief 分配 CUDA 内存
 * @param bytes 要分配的字节数
 * @return 分配的内存指针
 * @throws std::runtime_error 如果分配失败
 */
void* cuda_malloc(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t status = cudaMalloc(&ptr, bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed: " + 
                               std::string(cudaGetErrorString(status)));
    }
    return ptr;
}

/**
 * @brief 执行设备到设备的内存拷贝
 * @param dst 目标设备指针
 * @param src 源设备指针
 * @param bytes 要拷贝的字节数
 * @throws std::runtime_error 如果拷贝失败
 */
void cuda_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    cudaError_t status = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA device-to-device memcpy failed: " + 
                               std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief 执行设备到主机的内存拷贝
 * @param dst 目标主机指针
 * @param src 源设备指针
 * @param bytes 要拷贝的字节数
 * @throws std::runtime_error 如果拷贝失败
 */
void cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    cudaError_t status = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA device-to-host memcpy failed: " + 
                               std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief 执行主机到设备的内存拷贝
 * @param dst 目标设备指针
 * @param src 源主机指针
 * @param bytes 要拷贝的字节数
 * @throws std::runtime_error 如果拷贝失败
 */
void cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    cudaError_t status = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA host-to-device memcpy failed: " + 
                               std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief 创建共享指针管理的 CUDA 内存
 * @param bytes 要分配的字节数
 * @return 共享指针管理的 CUDA 内存
 */
std::shared_ptr<void> make_shared_cuda_memory(size_t bytes) {
    void* ptr = cuda_malloc(bytes);
    return std::shared_ptr<void>(ptr, [](void* p) { 
        if (p) {
            cudaError_t status = cudaFree(p);
            if (status != cudaSuccess) {
                // 记录错误但不抛出异常，因为在析构函数中抛出异常是危险的
                std::cerr << "Warning: CUDA free failed: " 
                         << cudaGetErrorString(status) << std::endl;
            }
        }
    });
}

/**
 * @brief 深拷贝 CUDA 内存
 * @param src 源设备指针
 * @param bytes 要拷贝的字节数
 * @return 共享指针管理的新 CUDA 内存
 */
std::shared_ptr<void> deep_copy_cuda_memory(const void* src, size_t bytes) {
    if (src == nullptr) {
        return nullptr;
    }
    
    void* dst = cuda_malloc(bytes);
    
    try {
        cuda_memcpy_d2d(dst, src, bytes);
    } catch (...) {
        cudaFree(dst);
        throw;
    }
    
    return std::shared_ptr<void>(dst, [](void* p) { 
        if (p) cudaFree(p); 
    });
}

/**
 * @brief 创建 CUDA Tensor 的深拷贝
 * @param src_tensor 源 Tensor
 * @return 共享指针管理的新 CUDA 内存
 * @throws std::runtime_error 如果源 Tensor 不是 CUDA Tensor
 */
std::shared_ptr<void> deep_copy_cuda_tensor(const Tensor& src_tensor) {
    if (!src_tensor.is_cuda()) {
        throw std::runtime_error("Source tensor is not a CUDA tensor");
    }
    if (src_tensor.raw_data() == nullptr) {
        return nullptr;
    }
    return deep_copy_cuda_memory(src_tensor.raw_data(), src_tensor.num_bytes());
}

}
}
}
}
