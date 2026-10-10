#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>  // std::runtime_error

#include "axono/core/tensor.h"

#ifdef AXONO_WITH_CUDA
#include "axono/ops/cuda/randn.h"
#include "axono/core/cuda/capture.h"
#include "axono/core/cuda/detail.h"
#include "axono/core/cuda/tensor/kernel.h"
#include "axono/core/cuda/tensor/transpose.h"
#endif

#include "axono/core/half.h"
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
      num_elements_(other.num_elements_),
      data_(other.data_) {
  // torch 语义: 拷贝 = 共享 storage (浅拷贝, 引用计数)。显式深拷贝用
  // Tensor(dtype, shape) + CopyFrom 或 to() 跨设备迁移。
  // 此前的深拷贝 copy ctor 曾导致: Module::add_weight 每参数复制一份
  // (2B 模型构建期显存翻倍 8.5→16.5GB)、to() 同设备复制等大量隐性浪费。
}
Tensor &Tensor::operator=(const Tensor &other) {
  if (this != &other) {
    // torch 语义: 拷贝赋值 = 共享 storage (浅拷贝)。
    data_.reset();
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    device_ = other.device_;
    num_elements_ = other.num_elements_;
    data_ = other.data_;
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

Tensor Tensor::FromBorrowed(DataType dtype, const Shape &shape,
                            const void *data) {
  Tensor t(dtype, shape);
  // no-op deleter: 借用内存, 析构不释放 (生命周期由调用方管理)。
  t.data_ = std::shared_ptr<void>(const_cast<void *>(data), [](void *) {});
  return t;
}

Tensor Tensor::to(const std::string &target_device) const {
  if (device_ == target_device) {
    // torch 语义: 同设备 to() 返回自身 (共享 storage, 零拷贝)。
    // 此前这里做了深拷贝, Linear/RMSNorm 等构建期 from_numpy→to(device) 双重
    // 分配直接翻倍显存占用 (2B 模型 build 阶段 16.5GB vs 8.8GB)。
    return *this;
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

  // 直接设备到设备写入 self 的 storage (self 的 data_ 已由构造/Resize 分配)。
  // (旧实现经 cpu 中转并整体替换 data_, 浅拷贝语义下 self_cpu 与 this 共享
  // storage 会误写共享块, 故重写为直拷。)
  if (is_cuda()) {
#ifdef AXONO_WITH_CUDA
    if (src.is_cuda()) {
      cuda::detail::cuda_memcpy_d2d(data<void *>(), src.data<void *>(),
                                    num_bytes());
    } else {
      cuda::detail::cuda_memcpy_h2d(data<void *>(), src.data<void *>(),
                                    num_bytes());
    }
#endif
  } else {
    if (src.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      cuda::detail::cuda_memcpy_d2h(data<void *>(), src.data<void *>(),
                                    num_bytes());
#endif
    } else {
      std::memcpy(data<void *>(), src.data<void *>(), num_bytes());
    }
  }
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
    case DataType::FLOAT16:
      oss << "float16";
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



namespace {

template <typename SrcT, typename DstT>
void CastLoop(const void *src, void *dst, size_t n) {
  const SrcT *s = static_cast<const SrcT *>(src);
  DstT *d = static_cast<DstT *>(dst);
  for (size_t i = 0; i < n; ++i) d[i] = static_cast<DstT>(s[i]);
}

}  // namespace

Tensor Tensor::CastTo(DataType target) const {
  if (dtype_ == target) return *this;  // 浅拷贝语义 (共享 storage)
  if (!data_) throw std::runtime_error("CastTo: empty tensor");
  Tensor dst(target, shape_, device_);
  dst.InitializeStorage();
  const size_t n = num_elements_;
  const void *sp = data_.get();
  void *dp = dst.data_.get();

#ifdef AXONO_WITH_CUDA
  if (is_cuda() && ((dtype_ == DataType::FLOAT16 &&
                     target == DataType::FLOAT32) ||
                    (dtype_ == DataType::FLOAT32 &&
                     target == DataType::FLOAT16))) {
    Status st = cuda::tensor::DispatchCastF16F32(dst, *this);
    if (st != Status::OK) throw std::runtime_error("CastTo CUDA failed");
    if (cuda::MaybeSync() != cudaSuccess)
      throw std::runtime_error("CastTo CUDA sync failed");
    return dst;
  }
#endif

  // fp16 CPU 路径: 手写 half 位模式转换
  if (dtype_ == DataType::FLOAT16 && target == DataType::FLOAT32) {
    const uint16_t *s = static_cast<const uint16_t *>(sp);
    float *d = static_cast<float *>(dp);
    for (size_t i = 0; i < n; ++i) d[i] = detail::Half2Float(s[i]);
    return dst;
  }
  if (dtype_ == DataType::FLOAT32 && target == DataType::FLOAT16) {
    const float *s = static_cast<const float *>(sp);
    uint16_t *d = static_cast<uint16_t *>(dp);
    for (size_t i = 0; i < n; ++i) d[i] = detail::Float2Half(s[i]);
    return dst;
  }

  // CPU 通用转换 (CUDA 上仅支持 fp16<->fp32 直转, 其余先回 CPU)
  if (is_cuda()) {
    throw std::runtime_error(
        "CastTo on CUDA: only FLOAT16<->FLOAT32 supported");
  }
  switch (dtype_) {
    case DataType::INT8: CastLoop<int8_t, float>(sp, dp, n);
      // 再 cast 到 target? 简化: 只支持转到 fp32
      if (target != DataType::FLOAT32) throw std::runtime_error("CastTo: unsupported");
      break;
    default: throw std::runtime_error("CastTo: unsupported source dtype");
  }
  return dst;
}

}  // namespace core
}  // namespace axono

