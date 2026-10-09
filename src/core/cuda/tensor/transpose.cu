#include <cuda_runtime.h>

#include "axono/core/tensor.h"
#include "axono/core/macros.h"
#include "axono/core/types.h"

namespace axono {
namespace core {
namespace cuda {
namespace tensor {

namespace {

// 通用 N-D 转置: 每个线程搬运一个元素。
// 先把线程线性 id 分解为源张量坐标, 交换 dim0/dim1 后映射到目标偏移。
template <typename T>
__global__ void TransposeKernel(const T* __restrict__ src,
                                T* __restrict__ dst,
                                const size_t* __restrict__ src_shape,
                                const size_t* __restrict__ src_stride,
                                const size_t* __restrict__ dst_stride,
                                int ndim, int dim0, int dim1,
                                size_t total) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // 线性 id -> 源坐标
    size_t coords[8];  // 支持最多 8 维
    size_t rem = idx;
    for (int i = 0; i < ndim; ++i) {
        coords[i] = rem / src_stride[i];
        rem %= src_stride[i];
    }

    // 交换维度
    size_t tmp = coords[dim0];
    coords[dim0] = coords[dim1];
    coords[dim1] = tmp;

    // 目标偏移
    size_t dst_idx = 0;
    for (int i = 0; i < ndim; ++i) {
        dst_idx += coords[i] * dst_stride[i];
    }
    dst[dst_idx] = src[idx];
}

template <typename T>
Status LaunchTransposeKernel(const Tensor& src, Tensor& dst, int dim0, int dim1) {
    const auto& shape = src.shape();
    const int ndim = static_cast<int>(shape.size());
    const size_t total = src.num_elements();
    if (total == 0) return Status::OK;

    // 源/目标 stride (行主序, 元素为单位)
    size_t h_src_stride[8] = {0};
    size_t h_dst_stride[8] = {0};
    h_src_stride[ndim - 1] = 1;
    h_dst_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        h_src_stride[i] = h_src_stride[i + 1] * shape[i + 1];
        h_dst_stride[i] = h_dst_stride[i + 1] * shape[dim0 == i + 1 ? dim1 : (dim1 == i + 1 ? dim0 : i + 1)];
    }
    // dst shape = src shape 且 dim0/dim1 已交换; 直接按交换后的形状算 stride
    // 上面的写法等价于: dst_shape[i] = shape[swap(i)], 但循环内表达式晦涩,
    // 这里用更直白的方式重算, 以正确性优先。
    size_t dst_shape[8];
    for (int i = 0; i < ndim; ++i) {
        int j = i;
        if (i == dim0) j = dim1;
        else if (i == dim1) j = dim0;
        dst_shape[i] = shape[j];
    }
    h_dst_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        h_dst_stride[i] = h_dst_stride[i + 1] * dst_shape[i + 1];
    }

    // 上传元数据到 device
    size_t* d_shape = nullptr;
    size_t* d_sstride = nullptr;
    size_t* d_dstride = nullptr;
    if (cudaMalloc(&d_shape, ndim * sizeof(size_t)) != cudaSuccess ||
        cudaMalloc(&d_sstride, ndim * sizeof(size_t)) != cudaSuccess ||
        cudaMalloc(&d_dstride, ndim * sizeof(size_t)) != cudaSuccess) {
        return Status::OUT_OF_MEMORY;
    }
    cudaMemcpy(d_shape, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sstride, h_src_stride, ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dstride, h_dst_stride, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);

    TransposeKernel<T><<<grid_size, block_size>>>(
        src.data<T>(), dst.data<T>(),
        d_shape, d_sstride, d_dstride,
        ndim, dim0, dim1, total);

    cudaError_t err = cudaGetLastError();
    cudaFree(d_shape);
    cudaFree(d_sstride);
    cudaFree(d_dstride);
    if (err != cudaSuccess) {
        return Status::DEVICE_ERROR;
    }
    return Status::OK;
}
}  // anonymous namespace

Status TransposeKernel(const Tensor& src, Tensor& dst, int dim0, int dim1) {
    if (!src.is_cuda() || !dst.is_cuda()) {
        return Status::DEVICE_MISMATCH;
    }
    if (src.dtype() != dst.dtype()) {
        return Status::UNSUPPORTED_TYPE;
    }

    const DataType dtype = src.dtype();
    switch (dtype) {
        case DataType::INT8:
            return LaunchTransposeKernel<int8_t>(src, dst, dim0, dim1);
        case DataType::INT16:
            return LaunchTransposeKernel<int16_t>(src, dst, dim0, dim1);
        case DataType::INT32:
            return LaunchTransposeKernel<int32_t>(src, dst, dim0, dim1);
        case DataType::INT64:
            return LaunchTransposeKernel<int64_t>(src, dst, dim0, dim1);
        case DataType::FLOAT32:
            return LaunchTransposeKernel<float>(src, dst, dim0, dim1);
        case DataType::FLOAT64:
            return LaunchTransposeKernel<double>(src, dst, dim0, dim1);
        case DataType::BOOLEAN:
            return LaunchTransposeKernel<bool>(src, dst, dim0, dim1);
        default:
            return Status::UNSUPPORTED_TYPE;
    }
}

}  // namespace tensor
}  // namespace cuda
}  // namespace core
}  // namespace axono
