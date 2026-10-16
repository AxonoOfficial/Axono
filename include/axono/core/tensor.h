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

// Tensor — Axono 核心张量类。
//
// 一块 [dtype + shape + device + 引用计数的存储] 的最小封装:
// - dtype:  FLOAT16 / FLOAT32 / FLOAT64 / INT8..64 / BOOLEAN
// - device: "cpu" 或 "cuda" / "cuda:<id>" (字符串承载, 便于扩展)
// - storage: shared_ptr<void> 统一管理 host / device 内存, 拷贝即共享
//
// 内存布局: C 连续 (row-major), 无 stride/视图机制 — 争取零开销。
// 跨设备: to()/to_() 走 H2D/D2D; CopyFrom() 设备间自动中转。
class Tensor {
 public:
  // 构造: 空 / 仅 dtype / dtype+shape / dtype+shape+device / 外部裸指针
  Tensor();
  explicit Tensor(DataType dtype);
  Tensor(DataType dtype, const Shape &shape);
  Tensor(DataType dtype, const Shape &shape, const std::string &device);
  Tensor(DataType dtype, const Shape &shape, void *data);

  // 拷贝 = 共享 storage (引用计数, 零数据拷贝), 传参零开销
  Tensor(const Tensor &other);
  Tensor &operator=(const Tensor &other);

  // 移动 = 接管 storage, 原张量置空
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;

  // 析构函数
  ~Tensor();

  // 工厂函数: 创建指定 dtype/shape 的新张量 (分配并清零存储)
  static Tensor Create(DataType dtype, const Shape &shape);
  // 同 dtype/shape/device 复制一个空壳 (分配新存储)
  static Tensor CreateLike(const Tensor &other);
  static Tensor FromData(DataType dtype, const Shape &shape, void *data);
  // 借用外部内存 (不拥有, 析构不释放) — 调用方须保证数据存活期覆盖 Tensor。
  // 适用 mmap 权重零拷贝加载: 从文件映射直接 H2D, 省去 host 中转拷贝。
  static Tensor FromBorrowed(DataType dtype, const Shape &shape,
                             const void *data);

  // ---- 基本信息 ----
  // 设备字符串 ("cpu" / "cuda:0"); dtype / shape / 元素数 / 字节数
  const std::string &device() const { return device_; }
  bool is_cuda() const { return device_.substr(0, 4) == "cuda"; }
  DataType dtype() const { return dtype_; }
  const Shape &shape() const { return shape_; }
  size_t ndim() const { return shape_.size(); }
  size_t num_elements() const { return num_elements_; }
  size_t num_bytes() const { return num_elements_ * GetDataTypeSize(dtype_); }
  bool is_contiguous() const { return true; }  // TODO

  // ---- 数据访问 ----
  // data<T>(): 按类型取底层指针 (host 或 device 指针, 视设备而定)
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

  // 从另一张量拷贝数据, 元素数须一致。设备间自动中转:
  // cpu→cuda / cuda→cpu 走 H2D/D2D, cuda→cuda 走 D2D。原地语义。
  Status CopyFrom(const Tensor &src);

  // dtype 类型转换 (返回新 Tensor, 同设备)。
  // 支持 FLOAT16<->FLOAT32 及各标量互转; FP16 常驻推理的关键入口。
  Tensor CastTo(DataType target) const;

  // 初始化数据存储 (按 dtype/shape/device 分配并清零)
  void InitializeStorage();

  // 形状操作 (均要求元素总数不变; Reshape 失败返回非 OK Status)
  Status Reshape(const Shape &new_shape);
  Status Resize(const Shape &new_shape);

  // 转置: 交换两个维度 (默认最后两维, 2D 即 a.T)
  Tensor Transpose(int dim0 = -2, int dim1 = -1);

  // 填充: 全 0 / 指定值 (按 value_size 字节模式)
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
