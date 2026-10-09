#include <stdexcept>
#include <algorithm>
#include <cstring>

#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace core {
namespace cpu {
namespace tensor {

namespace {
template <typename T>
void TransposeImpl(const Tensor& src, Tensor& dst, int dim0, int dim1) {
    const auto& src_shape = src.shape();
    const auto& dst_shape = dst.shape();
    const size_t total_elems = src.num_elements();

    if (total_elems == 0) return;
    if (dst.num_elements() != total_elems) throw std::logic_error("src/dst elem count mismatch");
    if (dim0 < 0 || dim0 >= (int)src_shape.size() || dim1 < 0 || dim1 >= (int)src_shape.size())
        throw std::out_of_range("dim index out of bound");

    std::vector<size_t> src_stride(src_shape.size(), 1);
    std::vector<size_t> dst_stride(dst_shape.size(), 1);
    for (int i = (int)src_shape.size() - 2; i >= 0; --i)
        src_stride[i] = src_stride[i+1] * src_shape[i+1];
    for (int i = (int)dst_shape.size() - 2; i >= 0; --i)
        dst_stride[i] = dst_stride[i+1] * dst_shape[i+1];

    const T* src_data = src.data<T>();
    T* dst_data = dst.data<T>();
    if (!src_data || !dst_data) throw std::runtime_error("null data pointer");

    for (size_t linear_idx = 0; linear_idx < total_elems; ++linear_idx) {
        std::vector<size_t> coords(src_shape.size(), 0);
        size_t rem = linear_idx;
        for (int i = 0; i < (int)src_shape.size(); ++i) {
            coords[i] = rem / src_stride[i];
            rem %= src_stride[i];
            if (coords[i] >= src_shape[i])
                throw std::out_of_range("src coord out of bound");
        }

        std::swap(coords[dim0], coords[dim1]);

        size_t dst_idx = 0;
        for (int i = 0; i < (int)dst_shape.size(); ++i) {
            if (coords[i] >= dst_shape[i])
                throw std::out_of_range("dst coord out of bound");
            dst_idx += coords[i] * dst_stride[i];
        }

        if (dst_idx >= dst.num_elements())
            throw std::out_of_range("dst linear index out of bound");

        dst_data[dst_idx] = src_data[linear_idx];
    }
}

Status DispatchDtype(const Tensor& src, Tensor& dst, int dim0, int dim1) {
    switch (src.dtype()) {
        case DataType::INT8:    TransposeImpl<int8_t>(src, dst, dim0, dim1); break;
        case DataType::INT16:   TransposeImpl<int16_t>(src, dst, dim0, dim1); break;
        case DataType::INT32:   TransposeImpl<int32_t>(src, dst, dim0, dim1); break;
        case DataType::INT64:   TransposeImpl<int64_t>(src, dst, dim0, dim1); break;
        case DataType::FLOAT32: TransposeImpl<float>(src, dst, dim0, dim1); break;
        case DataType::FLOAT64: TransposeImpl<double>(src, dst, dim0, dim1); break;
        case DataType::BOOLEAN: TransposeImpl<bool>(src, dst, dim0, dim1); break;
        default: return Status::UNSUPPORTED_TYPE;
    }
    return Status::OK;
}
} // anonymous namespace

Status TransposeKernel(const Tensor& src, Tensor& dst, int dim0, int dim1) {
    if (src.device() != "cpu" || dst.device() != "cpu") return Status::DEVICE_MISMATCH;
    if (src.dtype() != dst.dtype()) return Status::UNSUPPORTED_TYPE;
    if (src.num_elements() == 0) return Status::OK;
    return DispatchDtype(src, dst, dim0, dim1);
}

} // namespace tensor
} // namespace cpu
} // namespace core
} // namespace axono
