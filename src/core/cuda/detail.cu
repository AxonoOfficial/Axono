// Axono/src/core/cuda/detail.cu
#include <cuda_runtime.h>
#include <stdexcept>
#include "axono/core/tensor.h"

namespace axono {
namespace core {
namespace cuda {
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
    void* dev_ptr = nullptr;
    err = cudaMalloc(&dev_ptr, bytes);
    if (err != cudaSuccess) {
        throw std::bad_alloc();
    }

    // 初始化为0
    err = cudaMemset(dev_ptr, 0, bytes);
    if (err != cudaSuccess) {
        cudaFree(dev_ptr); // 清理已分配的内存
        throw std::runtime_error("CUDA memset failed: " + 
                               std::string(cudaGetErrorString(err)));
    }

    // 返回带CUDA释放器的智能指针
    return std::shared_ptr<void>(dev_ptr, [](void* ptr) {
        cudaFree(ptr);
    });
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
