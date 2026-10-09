#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>  // std::runtime_error

#include "axono/core/tensor.h"

#ifdef AXONO_WITH_CUDA
#include "axono/ops/cuda/randn.h"
#include "axono/core/cuda/detail.h"
#include "axono/core/cuda/tensor/kernel.h"
#include "axono/core/cuda/tensor/transpose.h"
#endif

#include "axono/ops/cpu/randn.h"
#include "axono/core/cpu/tensor/kernel.h"
#include "axono/core/types.h"
#include "axono/core/cpu/tensor/transpose.h"

namespace {
// 自定义删除器，用于 shared_ptr
struct FreeDeleter {
  void operator()(void *ptr) const { std::free(ptr); }
};
}  // namespace
namespace axono {
namespace core {

Tensor::Tensor() : dtype_(DataType::FLOAT32), num_elements_(0) {}

Tensor::Tensor(DataType dtype) : dtype_(dtype), num_elements_(0) {}

Tensor::Tensor(DataType dtype, const Shape &shape)
    : dtype_(dtype), shape_(shape) {
  num_elements_ = CalculateNumElements(shape_);
  InitializeStorage();
}

Tensor::Tensor(DataType dtype, const Shape &shape,
               const std::string &device)  // 设备在这里喵
    : dtype_(dtype), shape_(shape), device_(device) {
  num_elements_ = CalculateNumElements(shape_);
  InitializeStorage();
}

Tensor::Tensor(DataType dtype, const Shape &shape, void *data)
    : dtype_(dtype), shape_(shape) {
  num_elements_ = CalculateNumElements(shape_);
  data_ = std::shared_ptr<void>(data, FreeDeleter());
}

Tensor Tensor::randn(const std::vector<size_t> &shape, DataType dtype,
                     const std::string &device, float mean, float stddev) {
  Tensor out(dtype, shape, device);
  core::Context ctx;
  core::Status status;

  if (out.is_cuda()) {
#ifdef AXONO_WITH_CUDA
    status = ops::cuda::Randn(ctx, out, mean, stddev);
#else
    status = core::Status::DEVICE_ERROR;
#endif
  } else {
    status = ops::cpu::Randn(ctx, out, mean, stddev);
  }

  if (status != core::Status::OK) {
    throw std::runtime_error("Failed to generate randn tensor");
  }
  return out;
}

Tensor::Tensor(const Tensor &other)
    : dtype_(other.dtype_),
      shape_(other.shape_),
      device_(other.device_),
      num_elements_(other.num_elements_) {
  if (other.data_) {
    // 总是重新分配存储
    InitializeStorage();
    
    // 执行设备间拷贝
    if (is_cuda()) {
#ifdef AXONO_WITH_CUDA
      if (other.is_cuda()) {
        cuda::detail::cuda_memcpy_d2d(data_.get(), other.data_.get(),
                                      num_bytes());
      } else {
        cuda::detail::cuda_memcpy_h2d(data_.get(), other.data_.get(),
                                      num_bytes());
      }
#endif
    } else {
      if (other.is_cuda()) {
#ifdef AXONO_WITH_CUDA
        cuda::detail::cuda_memcpy_d2h(data_.get(), other.data_.get(),
                                      num_bytes());
#endif
      } else {
        std::memcpy(data_.get(), other.data_.get(), num_bytes());
      }
    }
  }
}
Tensor &Tensor::operator=(const Tensor &other) {
  if (this != &other) {
    // 清理旧数据
    data_.reset();
    
    // 更新所有成员变量
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    device_ = other.device_;  // 重要：更新设备信息！
    num_elements_ = other.num_elements_;
    
    if (other.data_) {
      // 重新初始化存储
      InitializeStorage();
      
      // 执行设备间正确的拷贝
      if (device_ == other.device_) {
        if (is_cuda()) {
#ifdef AXONO_WITH_CUDA
          cuda::detail::cuda_memcpy_d2d(data_.get(), other.data_.get(),
                                        num_bytes());
#endif
        } else {
          std::memcpy(data_.get(), other.data_.get(), num_bytes());
        }
      } else {
        // 跨设备拷贝
        if (other.is_cuda() && !is_cuda()) {
#ifdef AXONO_WITH_CUDA
          cuda::detail::cuda_memcpy_d2h(data_.get(), other.data_.get(),
                                        num_bytes());
#endif
        } else if (!other.is_cuda() && is_cuda()) {
#ifdef AXONO_WITH_CUDA
          cuda::detail::cuda_memcpy_h2d(data_.get(), other.data_.get(),
                                        num_bytes());
#endif
        } else {
          throw std::runtime_error("Unsupported device copy");
        }
      }
    }
  }
  return *this;
}

Tensor::Tensor(Tensor &&other) noexcept
    : dtype_(other.dtype_),
      shape_(std::move(other.shape_)),
      device_(std::move(other.device_)),
      num_elements_(other.num_elements_),
      data_(std::move(other.data_)) {
  other.dtype_ = DataType::FLOAT32;
  other.shape_.clear();
  other.num_elements_ = 0;
  other.device_ = "cpu";
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    dtype_ = other.dtype_;
    shape_ = std::move(other.shape_);
    num_elements_ = other.num_elements_;
    data_ = std::move(other.data_);

    other.num_elements_ = 0;
    other.shape_.clear();
  }
  return *this;
}

Tensor::~Tensor() = default;

Tensor Tensor::Create(DataType dtype, const Shape &shape) {
  return Tensor(dtype, shape);
}

Tensor Tensor::CreateLike(const Tensor &other) {
  return Tensor(other.dtype_, other.shape_);
}

Tensor Tensor::FromData(DataType dtype, const Shape &shape, void *data) {
  return Tensor(dtype, shape, data);
}

Tensor Tensor::to(const std::string &target_device) const {
  if (device_ == target_device) {
    Tensor copy(*this);
    return copy;
  }

  Tensor result(dtype_, shape_, target_device);

  // 执行内存拷贝
  if (is_cuda() && target_device.substr(0, 3) == "cpu") {
#ifdef AXONO_WITH_CUDA
    cuda::detail::cuda_memcpy_d2h(result.data<void *>(), data<void *>(),
                                  num_bytes());
#endif
  } else if (!is_cuda() && target_device.substr(0, 4) == "cuda") {
#ifdef AXONO_WITH_CUDA
    cuda::detail::cuda_memcpy_h2d(result.data<void *>(), data<void *>(),
                                  num_bytes());
#endif
  } else if (target_device.substr(0, 4) == "cuda") {
#ifdef AXONO_WITH_CUDA
    cuda::detail::cuda_memcpy_d2d(result.data<void *>(), data<void *>(),
                                  num_bytes());
#endif
  } else {
    std::memcpy(result.data<void *>(), data<void *>(), num_bytes());
  }

  return result;
}

Status Tensor::to_(const std::string &target_device) {
  // 原地迁移：通过to()创建新张量后交换内部数据
  Tensor temp = to(target_device);
  std::swap(*this, temp);
  return Status::OK;
}

Status Tensor::CopyFrom(const Tensor &src) {
  if (dtype_ != src.dtype_) return Status::UNSUPPORTED_TYPE;
  if (num_elements_ != src.num_elements_) return Status::SHAPE_MISMATCH;
  if (num_elements_ == 0) return Status::OK;

  // 统一中转: src -> cpu -> 逐字节写入 -> 拷回 self 所在设备
  Tensor tmp_cpu = src.to("cpu");
  Tensor self_cpu = this->to("cpu");
  std::memcpy(self_cpu.data<void *>(), tmp_cpu.data<void *>(), num_bytes());
  Tensor migrated = self_cpu.to(device_);

  // 用迁移结果的存储替换自身存储 (data_ 是 shared_ptr, 直接换)
  data_ = std::move(migrated.data_);
  return Status::OK;
}

void Tensor::InitializeStorage() {
  if (num_elements_ == 0) return;
  size_t bytes = num_bytes();
  if (bytes == 0) return;

  if (is_cuda()) {
#ifdef AXONO_WITH_CUDA
    data_ = cuda::detail::CudaAllocateStorage(bytes, device_);
#endif
  } else {
    // CPU HERE~
    void *ptr = std::malloc(bytes);
    if (ptr) {
      data_ = std::shared_ptr<void>(ptr, FreeDeleter());
      std::memset(ptr, 0, bytes);
    }
  }
}

Status Tensor::Reshape(const Shape &new_shape) {
  size_t new_num_elements = CalculateNumElements(new_shape);
  if (new_num_elements != num_elements_) {
    throw std::runtime_error("喵！Reshape要求形状相同的喵！");
  }
  shape_ = new_shape;
  return Status::OK;
}

Status Tensor::Resize(const Shape &new_shape) {
  size_t new_num_elements = CalculateNumElements(new_shape);
  if (new_num_elements != num_elements_) {
    shape_ = new_shape;
    num_elements_ = new_num_elements;
    InitializeStorage();
  } else {
    shape_ = new_shape;
  }
  return Status::OK;
}

Status Tensor::FillZero() {
  if (!data_) return Status::INVALID_ARGUMENT;

  switch (dtype_) {
    case DataType::INT8:
      if (this->is_cuda()) {
#ifdef AXONO_WITH_CUDA
        cuda::tensor::DispatchZero(*this);
        break;
#endif
      }
      cpu::tensor::TensorZeroKernel(data<int8_t>(), num_elements_);
      break;
    case DataType::INT16:
      if (this->is_cuda()) {
#ifdef AXONO_WITH_CUDA
        cuda::tensor::DispatchZero(*this);
        break;
#endif
      }
      cpu::tensor::TensorZeroKernel(data<int16_t>(), num_elements_);
      break;
    case DataType::INT32:
      if (this->is_cuda()) {
#ifdef AXONO_WITH_CUDA
        cuda::tensor::DispatchZero(*this);
        break;
#endif
      }
      cpu::tensor::TensorZeroKernel(data<int32_t>(), num_elements_);
      break;
    case DataType::INT64:
      if (this->is_cuda()) {
#ifdef AXONO_WITH_CUDA
        cuda::tensor::DispatchZero(*this);
        break;
#endif
      }
      cpu::tensor::TensorZeroKernel(data<int64_t>(), num_elements_);
      break;
    case DataType::FLOAT32:
      if (this->is_cuda()) {
#ifdef AXONO_WITH_CUDA
        cuda::tensor::DispatchZero(*this);
        break;
#endif
      }
      cpu::tensor::TensorZeroKernel(data<float>(), num_elements_);
      break;
    case DataType::FLOAT64:
      if (this->is_cuda()) {
#ifdef AXONO_WITH_CUDA
        cuda::tensor::DispatchZero(*this);
        break;
#endif
      }
      cpu::tensor::TensorZeroKernel(data<double>(), num_elements_);
      break;
    case DataType::BOOLEAN:
      if (this->is_cuda()) {
#ifdef AXONO_WITH_CUDA
        cuda::tensor::DispatchZero(*this);
        break;
#endif
      }
      cpu::tensor::TensorZeroKernel(data<bool>(), num_elements_);
      break;
    default:
      return Status::UNSUPPORTED_TYPE;
  }
  return Status::OK;
}

Status Tensor::Fill(void *value, size_t value_size) {
  if (!data_) return Status::INVALID_ARGUMENT;
  if (this->is_cuda()) {
#ifdef AXONO_WITH_CUDA
    return cuda::tensor::DispatchFill(*this, value, value_size);
#endif
  }
  return cpu::tensor::DispatchFill(*this, value, value_size);
}

bool Tensor::IsSameShape(const Tensor &other) const {
  return shape_ == other.shape_;
}

std::string Tensor::ToString() const {
  std::ostringstream oss;
  oss << "Tensor(shape=[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << shape_[i];
  }
  oss << "], dtype=";

  switch (dtype_) {
    case DataType::INT8:
      oss << "int8";
      break;
    case DataType::INT16:
      oss << "int16";
      break;
    case DataType::INT32:
      oss << "int32";
      break;
    case DataType::INT64:
      oss << "int64";
      break;
    case DataType::FLOAT32:
      oss << "float32";
      break;
    case DataType::FLOAT64:
      oss << "float64";
      break;
    case DataType::BOOLEAN:
      oss << "bool";
      break;
    default:
      oss << "unknown";
      break;
  }
  oss << ", device=" << device_ << ")";
  return oss.str();
}

Tensor Tensor::Transpose(int dim0, int dim1) {
    const int n_dim = static_cast<int>(this->ndim());
    dim0 = (dim0 < 0) ? (n_dim + dim0) : dim0;
    dim1 = (dim1 < 0) ? (n_dim + dim1) : dim1;

    if (dim0 < 0 || dim0 >= n_dim || dim1 < 0 || dim1 >= n_dim) {
        throw std::invalid_argument(
            "Transpose: invalid dims, ndim=" + std::to_string(n_dim) +
            ", dim0=" + std::to_string(dim0) + ", dim1=" + std::to_string(dim1)
        );
    }
    if (dim0 == dim1) {
        return *this;
    }

    Shape dst_shape = this->shape_;
    std::swap(dst_shape[dim0], dst_shape[dim1]);

    Tensor dst(this->dtype_, dst_shape, this->device_);
    dst.InitializeStorage();

    Status status;
    if (this->is_cuda()) {
#ifdef AXONO_WITH_CUDA
        status = cuda::tensor::TransposeKernel(*this, dst, dim0, dim1);
#else
        throw std::runtime_error("Transpose: CUDA not compiled, but tensor is on cuda");
#endif
    } else {
        status = cpu::tensor::TransposeKernel(*this, dst, dim0, dim1);
    }

    if (status != Status::OK) {
        throw std::runtime_error("Transpose failed, status=" + std::to_string(static_cast<int>(status)));
    }

    return dst;
}

}  // namespace core
}  // namespace axono
