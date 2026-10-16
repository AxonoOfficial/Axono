// Axono/include/axono/core/tensor.h
#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "macros.h"
#include "types.h"

// 遵循 lib, type
namespace axono {
namespace core {

class Tensor {
 public:
  // 构造函数
  Tensor();
  explicit Tensor(DataType dtype);
  Tensor(DataType dtype, const Shape &shape);
  Tensor(DataType dtype, const Shape &shape, const std::string &device);
  Tensor(DataType dtype, const Shape &shape, void *data);

  // 拷贝构造函数和赋值操作符
  Tensor(const Tensor &other);
  Tensor &operator=(const Tensor &other);

  // 移动构造函数和赋值操作符
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;

  // 析构函数
  ~Tensor();

  // 工厂函数: 创建指定 dtype/shape 的新张量
  static Tensor Create(DataType dtype, const Shape &shape);
  static Tensor CreateLike(const Tensor &other);
  static Tensor FromData(DataType dtype, const Shape &shape, void *data);
  // 借用外部内存 (不拥有, 析构不释放) — 调用方须保证数据存活期覆盖 Tensor。
  // 适用 mmap 权重零拷贝加载: 从文件映射直接 H2D, 省去 host 中转拷贝。
  static Tensor FromBorrowed(DataType dtype, const Shape &shape,
                             const void *data);

  // 基本信息
  const std::string &device() const { return device_; }
  bool is_cuda() const { return device_.substr(0, 4) == "cuda"; }
  DataType dtype() const { return dtype_; }
  const Shape &shape() const { return shape_; }
  size_t ndim() const { return shape_.size(); }
  size_t num_elements() const { return num_elements_; }
  size_t num_bytes() const { return num_elements_ * GetDataTypeSize(dtype_); }
  bool is_contiguous() const { return true; }  // TODO

  // 数据访问
  template <typename T>
  T *data() {
    if (is_cuda()) {
      // 确保返回的是设备指针
      return static_cast<T *>(data_.get());
    }
    return static_cast<T *>(data_.get());
  }

  template <typename T>
  const T *data() const {
    if (is_cuda()) return reinterpret_cast<const T *>(data_.get());
    return reinterpret_cast<const T *>(data_.get());
  }

  void *raw_data() { return data_.get(); }
  const void *raw_data() const { return data_.get(); }

  Tensor to(const std::string &target_device) const;
  Status to_(const std::string &target_device);

  // 从另一张量拷贝数据 (设备间自动中转), 元素数须一致
  Status CopyFrom(const Tensor &src);

  // 类型转换 (返回新 Tensor, 同设备)。支持 FLOAT16<->FLOAT32 及标量互转。
  Tensor CastTo(DataType target) const;

  // 初始化数据存储 (按 dtype/shape/device 分配并清零)
  void InitializeStorage();

  // 形状操作
  Status Reshape(const Shape &new_shape);
  Status Resize(const Shape &new_shape);

  // Transpose
  Tensor Transpose(int dim0 = -2, int dim1 = -1);

  // 填充操作
  Status FillZero();
  Status Fill(void *value, size_t value_size);

  // 标准正态分布随机初始化
  static Tensor randn(const std::vector<size_t> &shape,
                      DataType dtype = DataType::FLOAT32,
                      const std::string &device = "cpu", float mean = 0.0f,
                      float stddev = 1.0f);

  // 工具函数
  bool IsSameShape(const Tensor &other) const;
  std::string ToString() const;

  void *data() { return raw_data(); }
  const void *data() const { return raw_data(); }

 private:
  DataType dtype_ = DataType::FLOAT32;
  Shape shape_;
  std::string device_ =
      "cpu";  size_t num_elements_ = 0;
  std::shared_ptr<void> data_;
};

}  // namespace core
}  // namespace axono
