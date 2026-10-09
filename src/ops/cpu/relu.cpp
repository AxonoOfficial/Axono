#include <algorithm>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace ops {
namespace cpu {

template <typename T>
void ReluKernel(const T *input, T *output, size_t n) {
  for (size_t i = 0; i < n; ++i) output[i] = std::max(T(0), input[i]);
}
template <typename T>
void ReluInplaceKernel(T *data, size_t n) {
  for (size_t i = 0; i < n; ++i) data[i] = std::max(T(0), data[i]);
}

core::Status DispatchRelu(const core::Tensor &input, core::Tensor &output) {
  if (!input.IsSameShape(output)) return core::Status::SHAPE_MISMATCH;
  if (input.dtype() != output.dtype()) return core::Status::UNSUPPORTED_TYPE;

  const size_t n = input.num_elements();
  switch (input.dtype()) {
    case core::DataType::FLOAT32:
      ReluKernel(input.data<float>(), output.data<float>(), n);
      break;
    case core::DataType::FLOAT64:
      ReluKernel(input.data<double>(), output.data<double>(), n);
      break;
    case core::DataType::INT32:
      ReluKernel(input.data<int32_t>(), output.data<int32_t>(), n);
      break;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }
  return core::Status::OK;
}

core::Status DispatchReluInplace(core::Tensor &tensor) {
  const size_t n = tensor.num_elements();
  switch (tensor.dtype()) {
    case core::DataType::FLOAT32:
      ReluInplaceKernel(tensor.data<float>(), n);
      break;
    case core::DataType::FLOAT64:
      ReluInplaceKernel(tensor.data<double>(), n);
      break;
    case core::DataType::INT32:
      ReluInplaceKernel(tensor.data<int32_t>(), n);
      break;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }
  return core::Status::OK;
}

core::Status Relu(const core::Context &ctx, const core::Tensor &input,
                  core::Tensor &output) {
  (void)ctx;
  auto st = output.Resize(input.shape());
  if (st != core::Status::OK) return st;
  if (output.dtype() != input.dtype()) return core::Status::UNSUPPORTED_TYPE;
  return DispatchRelu(input, output);
}

core::Status ReluInplace(const core::Context &ctx, core::Tensor &tensor) {
  (void)ctx;
  return DispatchReluInplace(tensor);
}

}  // namespace cpu
}  // namespace ops
}  // namespace axono
