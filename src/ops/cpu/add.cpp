#include <cstring>

#include "axono/core/macros.h"
#include "axono/core/types.h"
#include "axono/core/tensor.h"

namespace axono {
namespace ops {
namespace cpu {

template <typename T>
AXONO_FORCE_INLINE void AddBroadcastKernel(const T *a, const T *b, T *out,
                                           size_t M, size_t K) {
  if (M * K < 16384) {
    for (size_t m = 0; m < M; ++m)
      for (size_t k = 0; k < K; ++k) out[m * K + k] = a[m * K + k] + b[k];
    return;
  }
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t m = 0; m < M; ++m) {
    for (size_t k = 0; k < K; ++k) {
      out[m * K + k] = a[m * K + k] + b[k];
    }
  }
}

template <typename T>
AXONO_FORCE_INLINE void AddKernel(const T *a, const T *b, T *result,
                                  size_t num_elements) {
  // 小规模直接算, 避免线程开销; 大规模 OpenMP 并行
  if (num_elements < 16384) {
    for (size_t i = 0; i < num_elements; ++i) result[i] = a[i] + b[i];
    return;
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < num_elements; ++i) {
    result[i] = a[i] + b[i];
  }
}

template <typename T>
AXONO_FORCE_INLINE void AddScalarKernel(const T *a, T scalar, T *result,
                                        size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    result[i] = a[i] + scalar;
  }
}

AXONO_FORCE_INLINE core::Status DispatchAdd(const core::Tensor &a,
                                            const core::Tensor &b,
                                            core::Tensor &result) {
  if (a.dtype() != b.dtype() || a.dtype() != result.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  auto a_shape = a.shape();
  auto b_shape = b.shape();

  if (a.IsSameShape(b) && a.IsSameShape(result)) {
    const size_t num = a.num_elements();
    switch (a.dtype()) {
      case core::DataType::FLOAT32:
        AddKernel(a.data<float>(), b.data<float>(), result.data<float>(), num);
        return core::Status::OK;
      case core::DataType::FLOAT64:
        AddKernel(a.data<double>(), b.data<double>(), result.data<double>(),
                  num);
        return core::Status::OK;
      case core::DataType::INT32:
        AddKernel(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
                  num);
        return core::Status::OK;
      case core::DataType::INT64:
        AddKernel(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
                  num);
        return core::Status::OK;
      default:
        return core::Status::UNSUPPORTED_TYPE;
    }
  }

  if (a_shape.size() == 2 && b_shape.size() == 1 && result.shape() == a_shape &&
      a_shape[1] == b_shape[0]) {
    const size_t M = a_shape[0];
    const size_t K = a_shape[1];
    switch (a.dtype()) {
      case core::DataType::FLOAT32:
        AddBroadcastKernel(a.data<float>(), b.data<float>(),
                           result.data<float>(), M, K);
        return core::Status::OK;
      case core::DataType::FLOAT64:
        AddBroadcastKernel(a.data<double>(), b.data<double>(),
                           result.data<double>(), M, K);
        return core::Status::OK;
      case core::DataType::INT32:
        AddBroadcastKernel(a.data<int32_t>(), b.data<int32_t>(),
                           result.data<int32_t>(), M, K);
        return core::Status::OK;
      case core::DataType::INT64:
        AddBroadcastKernel(a.data<int64_t>(), b.data<int64_t>(),
                           result.data<int64_t>(), M, K);
        return core::Status::OK;
      default:
        return core::Status::UNSUPPORTED_TYPE;
    }
  }

  return core::Status::SHAPE_MISMATCH;
}

AXONO_FORCE_INLINE core::Status DispatchAddScalar(const core::Tensor &a,
                                                  void *scalar,
                                                  size_t scalar_size,
                                                  core::Tensor &result) {
  auto num_elements = a.num_elements();

  if (!a.IsSameShape(result)) {
    return core::Status::SHAPE_MISMATCH;
  }

  if (a.dtype() != result.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  switch (a.dtype()) {
    case core::DataType::FLOAT32: {
      float scalar_value = 0.0f;
      if (scalar_size >= sizeof(float)) {
        memcpy(&scalar_value, scalar, sizeof(float));
      }
      AddScalarKernel(a.data<float>(), scalar_value, result.data<float>(),
                      num_elements);
      break;
    }
    case core::DataType::FLOAT64: {
      double scalar_value = 0.0;
      if (scalar_size >= sizeof(double)) {
        memcpy(&scalar_value, scalar, sizeof(double));
      }
      AddScalarKernel(a.data<double>(), scalar_value, result.data<double>(),
                      num_elements);
      break;
    }
    case core::DataType::INT32: {
      int32_t scalar_value = 0;
      if (scalar_size >= sizeof(int32_t)) {
        memcpy(&scalar_value, scalar, sizeof(int32_t));
      }
      AddScalarKernel(a.data<int32_t>(), scalar_value, result.data<int32_t>(),
                      num_elements);
      break;
    }
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }

  return core::Status::OK;
}

core::Status Add(const core::Context &ctx, const core::Tensor &a,
                 const core::Tensor &b, core::Tensor &result) {
  (void)ctx;

  if (a.dtype() != b.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  core::Status status = result.Resize(a.shape());
  if (status != core::Status::OK) {
    return status;
  }

  if (result.dtype() != a.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  return DispatchAdd(a, b, result);
}

core::Status AddInto(const core::Context &ctx, const core::Tensor &a,
                     const core::Tensor &b, core::Tensor &result) {
  (void)ctx;
  // 原地语义: result 与 a 可同一存储。要求三者在同一设备且 dtype 一致。
  if (a.dtype() != b.dtype() || a.dtype() != result.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }
  if (!result.IsSameShape(a) || !result.IsSameShape(b)) {
    return core::Status::SHAPE_MISMATCH;
  }
  // DispatchAdd 的核函数按元素读 a、b 并写 result, 允许三指针重叠,
  // 因此 result == a 时即为真正的原地累加 (result += b)。
  return DispatchAdd(a, b, result);
}

core::Status AddScalar(const core::Context &ctx, const core::Tensor &a,
                       void *scalar, size_t scalar_size, core::Tensor &result) {
  (void)ctx;

  core::Status status = result.Resize(a.shape());
  if (status != core::Status::OK) {
    return status;
  }

  if (result.dtype() != a.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  return DispatchAddScalar(a, scalar, scalar_size, result);
}

}  // namespace cpu
}  // namespace ops
}  // namespace axono
