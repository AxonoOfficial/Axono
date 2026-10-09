#include <random>
#include <vector>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"

namespace axono {
namespace ops {
namespace cpu {

namespace {
// 线程局部随机数生成器
thread_local std::mt19937 rng(std::random_device{}());

template <typename T>
void generate_normal(T* data, size_t num_elements, T mean, T stddev) {
  std::normal_distribution<T> dist(mean, stddev);
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = dist(rng);
  }
}
}  // namespace

core::Status Randn(const core::Context& ctx, core::Tensor& out, float mean,
                   float stddev) {
  (void)ctx;  // 未使用的上下文

  const size_t num_elements = out.num_elements();
  if (num_elements == 0) return core::Status::OK;

  switch (out.dtype()) {
    case core::DataType::FLOAT32:
      generate_normal(out.data<float>(), num_elements, static_cast<float>(mean),
                      static_cast<float>(stddev));
      break;
    case core::DataType::FLOAT64:
      generate_normal(out.data<double>(), num_elements,
                      static_cast<double>(mean), static_cast<double>(stddev));
      break;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }
  return core::Status::OK;
}

}  // namespace cpu
}  // namespace ops
}  // namespace axono
