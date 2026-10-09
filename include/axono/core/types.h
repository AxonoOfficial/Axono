#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace axono {
namespace core {

// 基础设备枚举
enum class DeviceType {
  CPU,
  CUDA,
};
// 基础数据类型枚举
enum class DataType { INT8, INT16, INT32, INT64, FLOAT32, FLOAT64, BOOLEAN };

// 获取数据类型大小
inline size_t GetDataTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::INT8:
      return 1;
    case DataType::INT16:
      return 2;
    case DataType::INT32:
      return 4;
    case DataType::INT64:
      return 8;
    case DataType::FLOAT32:
      return 4;
    case DataType::FLOAT64:
      return 8;
    case DataType::BOOLEAN:
      return 1;
    default:
      return 0;
  }
}

// 状态返回码
enum class Status {
  OK,
  INVALID_ARGUMENT,
  OUT_OF_MEMORY,
  UNSUPPORTED_TYPE,
  SHAPE_MISMATCH,
  INTERNAL_ERROR,
  DEVICE_ERROR,
  INVALID_PTR,
  DEVICE_MISMATCH
};

// 计算上下文
struct Context {
  int device_id = 0;
};

// Tensor 形状类型
using Shape = std::vector<size_t>;

// 计算总元素数量
inline size_t CalculateNumElements(const Shape &shape) {
  if (shape.empty()) return 0;
  size_t num_elements = 1;
  for (auto dim : shape) {
    num_elements *= dim;
  }
  return num_elements;
}

}  // namespace core
}  // namespace axono
