#include <curand_kernel.h>

#include "axono/core/cuda/stream.h"
#include <random>

#include "axono/core/types.h"
#include "axono/core/tensor.h"

namespace axono {
namespace ops {
namespace cuda {

template <typename T>
__global__ void RandnKernel(T* data, size_t num_elements, float mean, float stddev, unsigned int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float val = curand_normal(&state);
    data[idx] = static_cast<T>(mean + val * stddev);
}

template <typename T>
core::Status DispatchRandn(const core::Context& ctx, core::Tensor& out, float mean, float stddev) {
    (void)ctx;
    size_t num_elements = out.num_elements();
    if (num_elements == 0) return core::Status::OK;

    std::random_device rd;
    unsigned int seed = rd();

    // 启动核弹咯
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    RandnKernel<T><<<grid_size, block_size, 0, axono::core::cuda::AxonoCurrentStream()>>>(
        out.data<T>(), num_elements, mean, stddev, seed
    );

    return core::Status::OK;
}

core::Status Randn(const core::Context& ctx, core::Tensor& out, float mean, float stddev) {
    (void)ctx;
    if (out.is_cuda()) {
#ifdef AXONO_WITH_CUDA
        switch (out.dtype()) {
            case core::DataType::FLOAT32:
                return DispatchRandn<float>(ctx, out, mean, stddev);
            case core::DataType::FLOAT64:
                return DispatchRandn<double>(ctx, out, mean, stddev);
            default:
                return core::Status::UNSUPPORTED_TYPE;
        }
#else
        return core::Status::DEVICE_ERROR;
#endif
    } else {
        return core::Status::DEVICE_ERROR;
    }
}

template core::Status DispatchRandn<float>(const core::Context&, core::Tensor&, float, float);
template core::Status DispatchRandn<double>(const core::Context&, core::Tensor&, float, float);

} // namespace cuda
} // namespace ops
} // namespace axono
