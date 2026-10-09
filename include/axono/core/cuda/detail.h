// Axono/src/core/cuda/detail.h
#pragma once

#include <memory>
#include <string>

#include "axono/core/tensor.h"

namespace axono {
namespace core {
namespace cuda {
namespace detail {

// 分配 CUDA 设备内存，返回带 cudaFree 的 shared_ptr
std::shared_ptr<void> CudaAllocateStorage(size_t bytes,
                                          const std::string &device = "cuda:0");
std::shared_ptr<void> deep_copy_cuda_tensor(const Tensor &src_tensor);

void *cuda_malloc(size_t bytes);

void cuda_memcpy_d2d(void *dst, const void *src, size_t bytes);

void cuda_memcpy_d2h(void *dst, const void *src, size_t bytes);

void cuda_memcpy_h2d(void *dst, const void *src, size_t bytes);

std::shared_ptr<void> make_shared_cuda_memory(size_t bytes);

std::shared_ptr<void> deep_copy_cuda_memory(const void *src, size_t bytes);

}  // namespace detail
}  // namespace cuda
}  // namespace core
}  // namespace axono
