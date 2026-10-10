// Axono v0.2 — nanobind 绑定模块
// 取代 dev 分支的 pybind11_module.cpp + include/axono/pybind/*。
// 模块名保持 libaxono, Python 端 from libaxono import ... 不变。

#include <mutex>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "axono/core/module.h"
#include "axono/core/ops.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#ifdef AXONO_WITH_CUDA
#include "axono/core/cuda/capture.h"
#include "axono/core/cuda/capture_pool.h"
#include "axono/core/cuda/detail.h"
#include "axono/core/cuda/graph.h"
#include "axono/core/cuda/tensor/kernel.h"
#include "axono/ops/cuda/add.h"
#include "axono/ops/cuda/matmul.h"
#include "axono/ops/cuda/matmul_lt.h"
#include "axono/ops/cuda/relu.h"
#endif
#include "axono/ops/cpu/add.h"
#include "axono/ops/cpu/elementwise.h"
#include "axono/ops/cpu/fused.h"
#include "axono/ops/cpu/linear.h"
#include "axono/ops/cpu/llm.h"
#include "axono/ops/cpu/sequence.h"
#include "axono/ops/cpu/matmul.h"
#include "axono/ops/cpu/relu.h"
#ifdef AXONO_WITH_CUDA
#include "axono/ops/cuda/elementwise.h"
#include "axono/ops/cuda/fused.h"
#include "axono/ops/cuda/linear.h"
#include "axono/ops/cuda/llm.h"
#include "axono/ops/cuda/sequence.h"
#endif

namespace nb = nanobind;
using namespace axono;

// ---------------------------------------------------------------------------
// 工具: 根据 Tensor 构造 numpy ndarray 视图 (CPU 共享内存 / CUDA 拷回主机)
// ---------------------------------------------------------------------------
namespace {

template <typename T>
nb::ndarray<nb::numpy, T, nb::c_contig> tensor_to_ndarray(
    core::Tensor &self, nb::handle keep_alive) {
  // 注意: shape() 返回的 vector 绑定在 self 上, 需拷贝一份,
  // 否则 ndarray 持有的 shape 指针会随临时对象析构而悬空。
  std::vector<size_t> shp = self.shape();
  const size_t n = self.num_elements();
  if (self.is_cuda()) {
#ifdef AXONO_WITH_CUDA
    // CUDA: 拷回主机内存 (独立副本)
    auto host = std::make_unique<T[]>(n);
    auto status = core::cuda::tensor::TensorReadKernel(self.data<T>(),
                                                       host.get(), n);
    if (status != core::Status::OK) {
      throw std::runtime_error("CUDA 读取数据失败, 状态代码: " +
                               std::to_string(static_cast<int>(status)));
    }
    T *data = host.release();
    // capsule 管理临时主机内存; keep_alive 引用 Tensor 对象本身,
    // 确保 ndarray 存活期间底层 shape 拷贝与 CUDA 上下文有效
    nb::capsule free_when_done(data, [](void *ptr) noexcept {
      delete[] static_cast<T *>(ptr);
    });
    return nb::ndarray<nb::numpy, T, nb::c_contig>(
        /*data=*/data, /*ndim=*/shp.size(), /*shape=*/shp.data(),
        /*owner=*/free_when_done);
#else
    (void)keep_alive;
    throw std::runtime_error("本构建未启用 CUDA 支持");
#endif
  }
  // CPU: 共享内存视图。owner 传入 Tensor 自身的 handle,
  // 否则 nanobind 会复制数据, ndarray 与 Tensor 脱钩 (写入丢失)。
  T *data = self.data<T>();
  return nb::ndarray<nb::numpy, T, nb::c_contig>(
      /*data=*/data, /*ndim=*/shp.size(), /*shape=*/shp.data(),
      /*owner=*/keep_alive);
}

// fill 用的标量写入
core::Status fill_tensor(core::Tensor &t, void *value, size_t value_size) {
  return t.Fill(value, value_size);
}

core::Status check_device_match(const core::Tensor &a, const core::Tensor &b) {
  if (a.is_cuda() != b.is_cuda()) return core::Status::DEVICE_MISMATCH;
  return core::Status::OK;
}

// 逐元素算子按名分派 (避免在 CPU-only 构建里引用 CUDA 符号)
using EwBinaryFn = core::Status (*)(const core::Context &, const core::Tensor &,
                                    const core::Tensor &, core::Tensor &);
using EwUnaryFn = core::Status (*)(const core::Context &, const core::Tensor &,
                                   core::Tensor &);

EwBinaryFn EwBinCpu(const char *name) {
  std::string n(name);
  if (n == "sub") return &ops::cpu::Sub;
  if (n == "mul") return &ops::cpu::Mul;
  if (n == "div") return &ops::cpu::Div;
  if (n == "pow") return &ops::cpu::Pow;
  if (n == "maximum") return &ops::cpu::Maximum;
  if (n == "minimum") return &ops::cpu::Minimum;
  return nullptr;
}

EwUnaryFn EwUnaryCpu(const char *name) {
  std::string n(name);
  if (n == "neg") return &ops::cpu::Neg;
  if (n == "abs") return &ops::cpu::Abs;
  if (n == "exp") return &ops::cpu::Exp;
  if (n == "log") return &ops::cpu::Log;
  if (n == "sqrt") return &ops::cpu::Sqrt;
  if (n == "sigmoid") return &ops::cpu::Sigmoid;
  if (n == "tanh") return &ops::cpu::Tanh;
  if (n == "sin") return &ops::cpu::Sin;
  if (n == "cos") return &ops::cpu::Cos;
  if (n == "rsqrt") return &ops::cpu::Rsqrt;
  if (n == "square") return &ops::cpu::Square;
  if (n == "reciprocal") return &ops::cpu::Reciprocal;
  if (n == "sign") return &ops::cpu::Sign;
  if (n == "floor") return &ops::cpu::Floor;
  if (n == "ceil") return &ops::cpu::Ceil;
  if (n == "round") return &ops::cpu::Round;
  return nullptr;
}

#ifdef AXONO_WITH_CUDA
EwBinaryFn EwBinCuda(const char *name) {
  std::string n(name);
  if (n == "sub") return &ops::cuda::Sub;
  if (n == "mul") return &ops::cuda::Mul;
  if (n == "div") return &ops::cuda::Div;
  if (n == "pow") return &ops::cuda::Pow;
  if (n == "maximum") return &ops::cuda::Maximum;
  if (n == "minimum") return &ops::cuda::Minimum;
  return nullptr;
}

EwUnaryFn EwUnaryCuda(const char *name) {
  std::string n(name);
  if (n == "neg") return &ops::cuda::Neg;
  if (n == "abs") return &ops::cuda::Abs;
  if (n == "exp") return &ops::cuda::Exp;
  if (n == "log") return &ops::cuda::Log;
  if (n == "sqrt") return &ops::cuda::Sqrt;
  if (n == "sigmoid") return &ops::cuda::Sigmoid;
  if (n == "tanh") return &ops::cuda::Tanh;
  if (n == "sin") return &ops::cuda::Sin;
  if (n == "cos") return &ops::cuda::Cos;
  if (n == "rsqrt") return &ops::cuda::Rsqrt;
  if (n == "square") return &ops::cuda::Square;
  if (n == "reciprocal") return &ops::cuda::Reciprocal;
  if (n == "sign") return &ops::cuda::Sign;
  if (n == "floor") return &ops::cuda::Floor;
  if (n == "ceil") return &ops::cuda::Ceil;
  if (n == "round") return &ops::cuda::Round;
  return nullptr;
}
#endif

}  // namespace

// ---------------------------------------------------------------------------
// 模块定义
// ---------------------------------------------------------------------------
NB_MODULE(libaxono, m) {
  m.doc() = "Axono Library (v0.2, nanobind)";

  // ---- 枚举 ----
  nb::enum_<core::DataType>(m, "DataType")
      .value("INT8", core::DataType::INT8)
      .value("INT16", core::DataType::INT16)
      .value("INT32", core::DataType::INT32)
      .value("INT64", core::DataType::INT64)
      .value("FLOAT32", core::DataType::FLOAT32)
      .value("FLOAT64", core::DataType::FLOAT64)
      .value("BOOLEAN", core::DataType::BOOLEAN);

  nb::enum_<core::Status>(m, "Status")
      .value("OK", core::Status::OK)
      .value("INVALID_ARGUMENT", core::Status::INVALID_ARGUMENT)
      .value("OUT_OF_MEMORY", core::Status::OUT_OF_MEMORY)
      .value("UNSUPPORTED_TYPE", core::Status::UNSUPPORTED_TYPE)
      .value("SHAPE_MISMATCH", core::Status::SHAPE_MISMATCH)
      .value("INTERNAL_ERROR", core::Status::INTERNAL_ERROR)
      .value("DEVICE_ERROR", core::Status::DEVICE_ERROR)
      .value("DEVICE_MISMATCH", core::Status::DEVICE_MISMATCH);

  // ---- Tensor ----
  nb::class_<core::Tensor>(m, "Tensor")
      .def(nb::init<>())
      .def(nb::init<core::DataType>())
      .def(nb::init<core::DataType, const core::Shape &>(),
           nb::arg("dtype"), nb::arg("shape"))
      .def(nb::init<core::DataType, const core::Shape &,
                    const std::string &>(),
           nb::arg("dtype"), nb::arg("shape"), nb::arg("device"))
      .def_static("randn", &core::Tensor::randn, nb::arg("shape"),
                  nb::arg("dtype") = core::DataType::FLOAT32,
                  nb::arg("device") = "cpu", nb::arg("mean") = 0.0f,
                  nb::arg("stddev") = 1.0f)
      .def_static("create_like", &core::Tensor::CreateLike)
      .def("to", [](const core::Tensor &self, const std::string &device) {
             return self.to(device);
           }, nb::arg("device"))
      .def("copy_from",
           [](core::Tensor &self, const core::Tensor &src) {
             core::Status st = self.CopyFrom(src);
             if (st != core::Status::OK)
               throw std::runtime_error("copy_from 失败, 错误代码: " +
                                        std::to_string(static_cast<int>(st)));
           }, nb::arg("src"))
      .def("to_", &core::Tensor::to_, nb::arg("device"))
      .def("transpose", &core::Tensor::Transpose,
           nb::arg("dim0") = -2, nb::arg("dim1") = -1)
      .def("reshape", &core::Tensor::Reshape)
      .def("resize", &core::Tensor::Resize)
      .def("fill_zero", &core::Tensor::FillZero)
      .def("fill", [](core::Tensor &self, nb::object value) {
             // 按 dtype 分派标量填充
             switch (self.dtype()) {
               case core::DataType::INT8: {
                 int8_t v = nb::cast<int8_t>(value);
                 return fill_tensor(self, &v, sizeof(v));
               }
               case core::DataType::INT16: {
                 int16_t v = nb::cast<int16_t>(value);
                 return fill_tensor(self, &v, sizeof(v));
               }
               case core::DataType::INT32: {
                 int32_t v = nb::cast<int32_t>(value);
                 return fill_tensor(self, &v, sizeof(v));
               }
               case core::DataType::INT64: {
                 int64_t v = nb::cast<int64_t>(value);
                 return fill_tensor(self, &v, sizeof(v));
               }
               case core::DataType::FLOAT32: {
                 float v = nb::cast<float>(value);
                 return fill_tensor(self, &v, sizeof(v));
               }
               case core::DataType::FLOAT64: {
                 double v = nb::cast<double>(value);
                 return fill_tensor(self, &v, sizeof(v));
               }
               case core::DataType::BOOLEAN: {
                 bool v = nb::cast<bool>(value);
                 return fill_tensor(self, &v, sizeof(v));
               }
               default:
                 return core::Status::UNSUPPORTED_TYPE;
             }
           }, nb::arg("value"))
      .def("is_same_shape", &core::Tensor::IsSameShape)
      .def_prop_ro("is_cuda", &core::Tensor::is_cuda)
      .def_prop_ro("device", &core::Tensor::device)
      .def_prop_ro("dtype", &core::Tensor::dtype)
      .def_prop_ro("shape", [](const core::Tensor &self) {
             const auto &s = self.shape();
             return std::vector<size_t>(s.begin(), s.end());
           })
      .def_prop_ro("ndim", &core::Tensor::ndim)
      .def_prop_ro("num_elements", &core::Tensor::num_elements)
      .def_prop_ro("num_bytes", &core::Tensor::num_bytes)
      .def("data_int8", [](nb::object &self) {
             return tensor_to_ndarray<int8_t>(nb::cast<core::Tensor &>(self), self);
           })
      .def("data_int16", [](nb::object &self) {
             return tensor_to_ndarray<int16_t>(nb::cast<core::Tensor &>(self), self);
           })
      .def("data_int32", [](nb::object &self) {
             return tensor_to_ndarray<int32_t>(nb::cast<core::Tensor &>(self), self);
           })
      .def("data_int64", [](nb::object &self) {
             return tensor_to_ndarray<int64_t>(nb::cast<core::Tensor &>(self), self);
           })
      .def("data_float32", [](nb::object &self) {
             return tensor_to_ndarray<float>(nb::cast<core::Tensor &>(self), self);
           })
      .def("data_float64", [](nb::object &self) {
             return tensor_to_ndarray<double>(nb::cast<core::Tensor &>(self), self);
           })
      .def("data_bool", [](nb::object &self) {
             return tensor_to_ndarray<bool>(nb::cast<core::Tensor &>(self), self);
           })
      .def("__repr__", &core::Tensor::ToString)
      .def("__str__", &core::Tensor::ToString)
      .def("__copy__", [](const core::Tensor &self) {
             return core::Tensor(self.dtype(), self.shape(), self.device());
           })
      .def("__deepcopy__", [](const core::Tensor &self, nb::dict) {
             return self.to(self.device());
           });

  // ---- Module (nn 权重容器) ----
  nb::class_<core::Module>(m, "Module")
      .def(nb::init<>())
      .def("add_weight", &core::Module::add_weight, nb::arg("name"),
           nb::arg("weight"))
      .def("get_weight", &core::Module::get_weight, nb::arg("name"),
           nb::rv_policy::reference_internal)
      .def("weights", [](core::Module &self) {
             std::vector<std::string> names;
             for (const auto &kv : self.weights()) names.push_back(kv.first);
             return names;
           }, nb::rv_policy::reference_internal);

  // ---- 算子 (自由函数) ----
#ifdef AXONO_WITH_CUDA
  // 缓存池诊断 (显存占用排查)
  m.def("cached_bytes",
        []() { return axono::core::cuda::GetCachedBytes(); });
  m.def("set_cache_budget", [](size_t bytes) {
    axono::core::cuda::SetCacheBudget(bytes);
  });
#endif
  m.def("add", [](const core::Tensor &a, const core::Tensor &b) {
    if (check_device_match(a, b) != core::Status::OK)
      throw std::runtime_error("add: 输入张量不在同一设备上");
    if (a.dtype() != b.dtype())
      throw std::runtime_error("add: 输入张量数据类型不一致");
    core::Tensor result(a.dtype(), a.shape(), a.device());
    core::Status st;
    if (a.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = ops::cuda::Add(core::Context(), a, b, result);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = ops::cpu::Add(core::Context(), a, b, result);
    }
    if (st != core::Status::OK)
      throw std::runtime_error("add 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
    return result;
  }, nb::arg("a"), nb::arg("b"));

  m.def("matmul", [](const core::Tensor &a, const core::Tensor &b) {
    if (check_device_match(a, b) != core::Status::OK)
      throw std::runtime_error("matmul: 输入张量不在同一设备上");
    if (a.dtype() != b.dtype())
      throw std::runtime_error("matmul: 输入张量数据类型不一致");
    if (a.ndim() != 2 || b.ndim() != 2)
      throw std::runtime_error("matmul: 目前仅支持二维矩阵");
    if (a.shape()[1] != b.shape()[0])
      throw std::runtime_error("matmul: 形状不匹配");
    core::Tensor result(a.dtype(),
                        std::vector<size_t>{a.shape()[0], b.shape()[1]},
                        a.device());
    core::Status st;
    if (a.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = ops::cuda::MatMul(core::Context(), a, b, result);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = ops::cpu::MatMul(core::Context(), a, b, result);
    }
    if (st != core::Status::OK)
      throw std::runtime_error("matmul 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
    return result;
  }, nb::arg("a"), nb::arg("b"));

  // add_into: result = a + b, result 可与 a/b 同存储 (真 inplace)。
  m.def("add_into",
        [](const core::Tensor &a, const core::Tensor &b, core::Tensor &result) {
          if (check_device_match(a, b) != core::Status::OK ||
              check_device_match(a, result) != core::Status::OK)
            throw std::runtime_error("add_into: 输入张量不在同一设备上");
          if (a.dtype() != b.dtype() || a.dtype() != result.dtype())
            throw std::runtime_error("add_into: 数据类型不一致");
          if (!a.IsSameShape(result) || !b.IsSameShape(result))
            throw std::runtime_error("add_into: 形状不匹配");
          core::Status st;
          if (a.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::Add(core::Context(), a, b, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::Add(core::Context(), a, b, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("add_into 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
        },
        nb::arg("a"), nb::arg("b"), nb::arg("out"));

  // add_: a += b (真原地, 零分配)
  m.def("add_", [](core::Tensor &a, const core::Tensor &b) {
    if (check_device_match(a, b) != core::Status::OK)
      throw std::runtime_error("add_: 输入张量不在同一设备上");
    if (a.dtype() != b.dtype())
      throw std::runtime_error("add_: 数据类型不一致");
    if (!a.IsSameShape(b))
      throw std::runtime_error("add_: 形状不匹配");
    core::Status st;
    if (a.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = ops::cuda::Add(core::Context(), a, b, a);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = ops::cpu::Add(core::Context(), a, b, a);
    }
    if (st != core::Status::OK)
      throw std::runtime_error("add_ 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
  }, nb::arg("a"), nb::arg("b"), nb::sig("def add_(a, b) -> None"));

  // matmul_into: result = a @ b, out 参数版
  m.def("matmul_into",
        [](const core::Tensor &a, const core::Tensor &b, core::Tensor &result) {
          if (check_device_match(a, b) != core::Status::OK ||
              check_device_match(a, result) != core::Status::OK)
            throw std::runtime_error("matmul_into: 输入张量不在同一设备上");
          if (a.dtype() != b.dtype() || a.dtype() != result.dtype())
            throw std::runtime_error("matmul_into: 数据类型不一致");
          if (a.ndim() != 2 || b.ndim() != 2)
            throw std::runtime_error("matmul_into: 目前仅支持二维矩阵");
          if (a.shape()[1] != b.shape()[0])
            throw std::runtime_error("matmul_into: 形状不匹配");
          if (result.shape()[0] != a.shape()[0] ||
              result.shape()[1] != b.shape()[1])
            throw std::runtime_error("matmul_into: out 形状不匹配");
          core::Status st;
          if (a.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::MatMul(core::Context(), a, b, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::MatMul(core::Context(), a, b, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("matmul_into 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
        },
        nb::arg("a"), nb::arg("b"), nb::arg("out"));

  // matmul_: a = a @ b (cuBLAS 无法别名读写, 经一次临时后 swap,
  // 语义等同 torch 的 out-place + 赋值, 但不新增 Python 对象)
  m.def("matmul_", [](core::Tensor &a, const core::Tensor &b) {
    if (check_device_match(a, b) != core::Status::OK)
      throw std::runtime_error("matmul_: 输入张量不在同一设备上");
    if (a.dtype() != b.dtype())
      throw std::runtime_error("matmul_: 数据类型不一致");
    if (a.ndim() != 2 || b.ndim() != 2)
      throw std::runtime_error("matmul_: 目前仅支持二维矩阵");
    if (a.shape()[1] != b.shape()[0])
      throw std::runtime_error("matmul_: 形状不匹配");
    core::Tensor result(a.dtype(),
                        std::vector<size_t>{a.shape()[0], b.shape()[1]},
                        a.device());
    core::Status st;
    if (a.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = ops::cuda::MatMul(core::Context(), a, b, result);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = ops::cpu::MatMul(core::Context(), a, b, result);
    }
    if (st != core::Status::OK)
      throw std::runtime_error("matmul_ 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
    a = std::move(result);
  }, nb::arg("a"), nb::arg("b"), nb::sig("def matmul_(a, b) -> None"));

  // relu: 非原地, 返回新张量 (输入保证不变)
  m.def("relu", [](const core::Tensor &input) {
    core::Tensor result(input.dtype(), input.shape(), input.device());
    core::Status st;
    if (input.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = ops::cuda::Relu(core::Context(), input, result);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = ops::cpu::Relu(core::Context(), input, result);
    }
    if (st != core::Status::OK)
      throw std::runtime_error("relu 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
    return result;
  }, nb::arg("x"));

  // relu_: 真原地 (直接在输入存储上执行 kernel, 无任何拷贝/临时)
  m.def("relu_", [](core::Tensor &input) {
    core::Status st;
    if (input.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = ops::cuda::ReluInplace(core::Context(), input);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = ops::cpu::ReluInplace(core::Context(), input);
    }
    if (st != core::Status::OK)
      throw std::runtime_error("relu_ 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
  }, nb::arg("x"), nb::sig("def relu_(x) -> None"));

  // ---- 逐元素算子 (sub/mul/div/neg/abs/exp/log/sqrt/sigmoid/tanh) ----
  // 统一 helper: 设备分派 (CUDA 优先, 回退 CPU)
  auto ew_binary = [&](const core::Tensor &a, const core::Tensor &b,
                       const char *name) {
    if (check_device_match(a, b) != core::Status::OK)
      throw std::runtime_error(std::string(name) + ": 输入张量不在同一设备上");
    if (a.dtype() != b.dtype())
      throw std::runtime_error(std::string(name) + ": 数据类型不一致");
    core::Tensor result(a.dtype(), a.shape(), a.device());
    core::Status st;
    if (a.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = EwBinCuda(name)(core::Context(), a, b, result);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = EwBinCpu(name)(core::Context(), a, b, result);
    }
    if (st != core::Status::OK)
      throw std::runtime_error(std::string(name) + " 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
    return result;
  };
  auto ew_unary = [&](const core::Tensor &x, const char *name) {
    core::Tensor result(x.dtype(), x.shape(), x.device());
    core::Status st;
    if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = EwUnaryCuda(name)(core::Context(), x, result);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = EwUnaryCpu(name)(core::Context(), x, result);
    }
    if (st != core::Status::OK)
      throw std::runtime_error(std::string(name) + " 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
    return result;
  };

  m.def("sub", [&](const core::Tensor &a, const core::Tensor &b) {
         return ew_binary(a, b, "sub");
       }, nb::arg("a"), nb::arg("b"));
  m.def("mul", [&](const core::Tensor &a, const core::Tensor &b) {
         return ew_binary(a, b, "mul");
       }, nb::arg("a"), nb::arg("b"));
  m.def("div", [&](const core::Tensor &a, const core::Tensor &b) {
         return ew_binary(a, b, "div");
       }, nb::arg("a"), nb::arg("b"));
  m.def("neg", [&](const core::Tensor &x) { return ew_unary(x, "neg"); },
       nb::arg("x"));
  m.def("abs", [&](const core::Tensor &x) { return ew_unary(x, "abs"); },
       nb::arg("x"));
  m.def("exp", [&](const core::Tensor &x) { return ew_unary(x, "exp"); },
       nb::arg("x"));
  m.def("log", [&](const core::Tensor &x) { return ew_unary(x, "log"); },
       nb::arg("x"));
  m.def("sqrt", [&](const core::Tensor &x) { return ew_unary(x, "sqrt"); },
       nb::arg("x"));
  m.def("sigmoid", [&](const core::Tensor &x) {
         return ew_unary(x, "sigmoid");
       }, nb::arg("x"));
  m.def("tanh", [&](const core::Tensor &x) {
         return ew_unary(x, "tanh");
       }, nb::arg("x"));
  m.def("sin", [&](const core::Tensor &x) { return ew_unary(x, "sin"); },
       nb::arg("x"));
  m.def("cos", [&](const core::Tensor &x) { return ew_unary(x, "cos"); },
       nb::arg("x"));
  m.def("rsqrt", [&](const core::Tensor &x) { return ew_unary(x, "rsqrt"); },
       nb::arg("x"));
  m.def("square", [&](const core::Tensor &x) {
         return ew_unary(x, "square");
       }, nb::arg("x"));
  m.def("reciprocal", [&](const core::Tensor &x) {
         return ew_unary(x, "reciprocal");
       }, nb::arg("x"));
  m.def("sign", [&](const core::Tensor &x) { return ew_unary(x, "sign"); },
       nb::arg("x"));
  m.def("floor", [&](const core::Tensor &x) { return ew_unary(x, "floor"); },
       nb::arg("x"));
  m.def("ceil", [&](const core::Tensor &x) { return ew_unary(x, "ceil"); },
       nb::arg("x"));
  m.def("round", [&](const core::Tensor &x) { return ew_unary(x, "round"); },
       nb::arg("x"));
  m.def("pow", [&](const core::Tensor &a, const core::Tensor &b) {
         return ew_binary(a, b, "pow");
       }, nb::arg("a"), nb::arg("b"));
  m.def("maximum", [&](const core::Tensor &a, const core::Tensor &b) {
         return ew_binary(a, b, "maximum");
       }, nb::arg("a"), nb::arg("b"));
  m.def("minimum", [&](const core::Tensor &a, const core::Tensor &b) {
         return ew_binary(a, b, "minimum");
       }, nb::arg("a"), nb::arg("b"));

  // ---- 单算子融合 Linear: y = x @ W^T + bias ----
  auto linear_impl = [&](const core::Tensor &x, const core::Tensor &weight,
                         const core::Tensor &bias) {
    if (x.is_cuda() != weight.is_cuda() ||
        (bias.num_elements() != 0 && bias.is_cuda() != x.is_cuda()))
      throw std::runtime_error("linear: 输入张量不在同一设备上");
    if (x.dtype() != weight.dtype() ||
        (bias.num_elements() != 0 && bias.dtype() != x.dtype()))
      throw std::runtime_error("linear: 数据类型不一致");
    core::Tensor result(x.dtype(), std::vector<size_t>{x.shape()[0],
                                                       weight.shape()[0]},
                        x.device());
    core::Status st;
    if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = ops::cuda::Linear(core::Context(), x, weight, bias, result);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = ops::cpu::Linear(core::Context(), x, weight, bias, result);
    }
    if (st != core::Status::OK)
      throw std::runtime_error("linear 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
    return result;
  };
  m.def("linear",
        [&](const core::Tensor &x, const core::Tensor &weight,
            const core::Tensor &bias) { return linear_impl(x, weight, bias); },
        nb::arg("x"), nb::arg("weight"), nb::arg("bias"),
        nb::sig("def linear(x, weight, bias) -> Tensor"));
  m.def("linear_nobias",
        [&](const core::Tensor &x, const core::Tensor &weight) {
          core::Tensor empty;
          return linear_impl(x, weight, empty);
        },
        nb::arg("x"), nb::arg("weight"),
        nb::sig("def linear_nobias(x, weight) -> Tensor"));

  // ---- LLM 相关算子 ----
  auto llm_unary = [&](const core::Tensor &x, const char *name) {
    core::Tensor result(x.dtype(), x.shape(), x.device());
    core::Status st;
    if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      if (std::string(name) == "softmax")
        st = ops::cuda::Softmax(core::Context(), x, result);
      else if (std::string(name) == "log_softmax")
        st = ops::cuda::LogSoftmax(core::Context(), x, result);
      else if (std::string(name) == "gelu")
        st = ops::cuda::Gelu(core::Context(), x, result);
      else
        st = ops::cuda::Silu(core::Context(), x, result);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      if (std::string(name) == "softmax")
        st = ops::cpu::Softmax(core::Context(), x, result);
      else if (std::string(name) == "log_softmax")
        st = ops::cpu::LogSoftmax(core::Context(), x, result);
      else if (std::string(name) == "gelu")
        st = ops::cpu::Gelu(core::Context(), x, result);
      else
        st = ops::cpu::Silu(core::Context(), x, result);
    }
    if (st != core::Status::OK)
      throw std::runtime_error(std::string(name) + " 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
    return result;
  };
  m.def("softmax", [&](const core::Tensor &x) { return llm_unary(x, "softmax"); },
        nb::arg("x"), nb::sig("def softmax(x) -> Tensor"));
  m.def("log_softmax",
        [&](const core::Tensor &x) { return llm_unary(x, "log_softmax"); },
        nb::arg("x"), nb::sig("def log_softmax(x) -> Tensor"));
  m.def("gelu", [&](const core::Tensor &x) { return llm_unary(x, "gelu"); },
        nb::arg("x"), nb::sig("def gelu(x) -> Tensor"));
  m.def("gelu_tanh",
        [&](const core::Tensor &x) {
          core::Tensor result(x.dtype(), x.shape(), x.device());
          core::Status st;
          if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::GeluTanh(core::Context(), x, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::GeluTanh(core::Context(), x, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("gelu_tanh 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return result;
        },
        nb::arg("x"), nb::sig("def gelu_tanh(x) -> Tensor"));
  m.def("silu", [&](const core::Tensor &x) { return llm_unary(x, "silu"); },
        nb::arg("x"), nb::sig("def silu(x) -> Tensor"));

  auto layer_norm_impl = [&](const core::Tensor &x, const core::Tensor &weight,
                             const core::Tensor &bias, float eps) {
    if (weight.is_cuda() != x.is_cuda() || bias.is_cuda() != x.is_cuda())
      throw std::runtime_error("layer_norm: 输入张量不在同一设备上");
    if (weight.dtype() != x.dtype() || bias.dtype() != x.dtype())
      throw std::runtime_error("layer_norm: 数据类型不一致");
    core::Tensor result(x.dtype(), x.shape(), x.device());
    core::Status st;
    if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = ops::cuda::LayerNorm(core::Context(), x, weight, bias, eps, result);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = ops::cpu::LayerNorm(core::Context(), x, weight, bias, eps, result);
    }
    if (st != core::Status::OK)
      throw std::runtime_error("layer_norm 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
    return result;
  };
  m.def("layer_norm",
        [&](const core::Tensor &x, const core::Tensor &weight,
            const core::Tensor &bias, float eps) {
          return layer_norm_impl(x, weight, bias, eps);
        },
        nb::arg("x"), nb::arg("weight"), nb::arg("bias"), nb::arg("eps") = 1e-5f,
        nb::sig("def layer_norm(x, weight, bias, eps=1e-5) -> Tensor"));

  auto rms_norm_impl = [&](const core::Tensor &x, const core::Tensor &weight,
                           float eps) {
    if (weight.is_cuda() != x.is_cuda() || weight.dtype() != x.dtype())
      throw std::runtime_error("rms_norm: 输入不在同一设备或类型不一致");
    core::Tensor result(x.dtype(), x.shape(), x.device());
    core::Status st;
    if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
      st = ops::cuda::RmsNorm(core::Context(), x, weight, eps, result);
#else
      st = core::Status::DEVICE_ERROR;
#endif
    } else {
      st = ops::cpu::RmsNorm(core::Context(), x, weight, eps, result);
    }
    if (st != core::Status::OK)
      throw std::runtime_error("rms_norm 失败, 错误代码: " +
                               std::to_string(static_cast<int>(st)));
    return result;
  };
  m.def("rms_norm",
        [&](const core::Tensor &x, const core::Tensor &weight, float eps) {
          return rms_norm_impl(x, weight, eps);
        },
        nb::arg("x"), nb::arg("weight"), nb::arg("eps") = 1e-5f,
        nb::sig("def rms_norm(x, weight, eps=1e-5) -> Tensor"));

  // ---- 序列/LLM 结构算子 (embedding/rope/attention/concat/slice/argmax) ----
  m.def("embedding",
        [&](const core::Tensor &ids, const core::Tensor &table) {
          if (ids.is_cuda() != table.is_cuda())
            throw std::runtime_error("embedding: 输入张量不在同一设备上");
          std::vector<size_t> out_shape = ids.shape();
          out_shape.push_back(table.shape()[1]);
          core::Tensor result(table.dtype(), out_shape, table.device());
          core::Status st;
          if (ids.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::Embedding(core::Context(), ids, table, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::Embedding(core::Context(), ids, table, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("embedding 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return result;
        },
        nb::arg("ids"), nb::arg("table"),
        nb::sig("def embedding(ids, table) -> Tensor"));

  m.def("rope",
        [&](const core::Tensor &x, const core::Tensor &pos_ids, float theta) {
          core::Tensor result(x.dtype(), x.shape(), x.device());
          core::Status st;
          if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::Rope(core::Context(), x, pos_ids, theta, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::Rope(core::Context(), x, pos_ids, theta, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("rope 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return result;
        },
        nb::arg("x"), nb::arg("pos_ids"), nb::arg("theta"),
        nb::sig("def rope(x, pos_ids, theta) -> Tensor"));

  m.def("rope_with_cos_sin",
        [&](const core::Tensor &x, const core::Tensor &cos,
            const core::Tensor &sin) {
          core::Tensor result(x.dtype(), x.shape(), x.device());
          core::Status st;
          if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::RopeWithCosSin(core::Context(), x, cos, sin, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::RopeWithCosSin(core::Context(), x, cos, sin, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("rope_with_cos_sin 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return result;
        },
        nb::arg("x"), nb::arg("cos"), nb::arg("sin"),
        nb::sig("def rope_with_cos_sin(x, cos, sin) -> Tensor"));

  m.def("rope_thd",
        [&](const core::Tensor &x, const core::Tensor &pos,
            const core::Tensor &inv_freq, int t_sec, int h_sec, int w_sec) {
          core::Tensor result(x.dtype(), x.shape(), x.device());
          core::Status st;
          if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::RopeThd(core::Context(), x, pos, inv_freq, t_sec,
                                    h_sec, w_sec, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::RopeThd(core::Context(), x, pos, inv_freq, t_sec,
                                   h_sec, w_sec, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("rope_thd 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return result;
        },
        nb::arg("x"), nb::arg("pos"), nb::arg("inv_freq"), nb::arg("t_sec"),
        nb::arg("h_sec"), nb::arg("w_sec"),
        nb::sig("def rope_thd(x, pos, inv_freq, t_sec, h_sec, w_sec) -> Tensor"));

  m.def("silu_mul",
        [](const core::Tensor &gate, const core::Tensor &up) {
          core::Tensor result(gate.dtype(), gate.shape(), gate.device());
          core::Status st;
          if (gate.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::SiluMul(core::Context(), gate, up, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::SiluMul(core::Context(), gate, up, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("silu_mul 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return result;
        },
        nb::arg("gate"), nb::arg("up"),
        nb::sig("def silu_mul(gate, up) -> Tensor"));

  m.def("add_rms_norm",
        [](const core::Tensor &x, const core::Tensor &residual,
           const core::Tensor &weight, float eps) {
          core::Tensor y(x.dtype(), x.shape(), x.device());
          core::Tensor out(x.dtype(), x.shape(), x.device());
          core::Status st;
          if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::AddRmsNorm(core::Context(), x, residual, weight,
                                       eps, y, out);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::AddRmsNorm(core::Context(), x, residual, weight,
                                      eps, y, out);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("add_rms_norm 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return nb::make_tuple(y, out);
        },
        nb::arg("x"), nb::arg("residual"), nb::arg("weight"), nb::arg("eps"),
        nb::sig("def add_rms_norm(x, residual, weight, eps) -> (Tensor, Tensor)"));

  m.def("mrope_cos_sin",
        [](const core::Tensor &pos, const core::Tensor &inv_freq, int h_sec,
           int w_sec) {
          core::Tensor cos_out(core::DataType::FLOAT32, {}, pos.device());
          core::Tensor sin_out(core::DataType::FLOAT32, {}, pos.device());
          core::Status st;
          if (pos.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::MropeCosSin(core::Context(), pos, inv_freq, h_sec,
                                        w_sec, cos_out, sin_out);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::MropeCosSin(core::Context(), pos, inv_freq, h_sec,
                                       w_sec, cos_out, sin_out);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("mrope_cos_sin 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return nb::make_tuple(cos_out, sin_out);
        },
        nb::arg("pos"), nb::arg("inv_freq"), nb::arg("h_sec"),
        nb::arg("w_sec"),
        nb::sig("def mrope_cos_sin(pos, inv_freq, h_sec, w_sec) -> (Tensor, Tensor)"));

  m.def("scaled_dot_product_attention",
        [&](const core::Tensor &q, const core::Tensor &k,
            const core::Tensor &v, bool is_causal) {
          std::vector<size_t> out_shape = q.shape();
          core::Tensor result(q.dtype(), out_shape, q.device());
          core::Status st;
          if (q.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::ScaledDotProductAttention(core::Context(), q, k, v,
                                                      is_causal, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::ScaledDotProductAttention(core::Context(), q, k, v,
                                                     is_causal, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("attention 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return result;
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("is_causal") = false,
        nb::sig("def scaled_dot_product_attention(q, k, v, is_causal=False) -> Tensor"));

  m.def("concat",
        [&](const core::Tensor &a, const core::Tensor &b, int axis) {
          if (a.is_cuda() != b.is_cuda() || a.dtype() != b.dtype())
            throw std::runtime_error("concat: 输入不在同一设备或类型不一致");
          std::vector<size_t> shape = a.shape();
          const size_t nd = a.ndim();
          size_t ax = axis < 0 ? nd + static_cast<size_t>(axis)
                               : static_cast<size_t>(axis);
          shape[ax] += b.shape()[ax];
          core::Tensor result(a.dtype(), shape, a.device());
          core::Status st;
          if (a.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::Concat(core::Context(), a, b, axis, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::Concat(core::Context(), a, b, axis, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("concat 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return result;
        },
        nb::arg("a"), nb::arg("b"), nb::arg("axis"),
        nb::sig("def concat(a, b, axis) -> Tensor"));

  m.def("slice",
        [&](const core::Tensor &x, size_t axis, size_t start, size_t length) {
          std::vector<size_t> shape = x.shape();
          shape[axis] = length;
          core::Tensor result(x.dtype(), shape, x.device());
          core::Status st;
          if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::Slice(core::Context(), x, axis, start, length,
                                  result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::Slice(core::Context(), x, axis, start, length,
                                 result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("slice 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return result;
        },
        nb::arg("x"), nb::arg("axis"), nb::arg("start"), nb::arg("length"),
        nb::sig("def slice(x, axis, start, length) -> Tensor"));

  m.def("argmax",
        [&](const core::Tensor &x) {
          const size_t rows = x.num_elements() / x.shape().back();
          core::Tensor result(core::DataType::INT64, {rows}, x.device());
          core::Status st;
          if (x.is_cuda()) {
#ifdef AXONO_WITH_CUDA
            st = ops::cuda::ArgmaxLastDim(core::Context(), x, result);
#else
            st = core::Status::DEVICE_ERROR;
#endif
          } else {
            st = ops::cpu::ArgmaxLastDim(core::Context(), x, result);
          }
          if (st != core::Status::OK)
            throw std::runtime_error("argmax 失败, 错误代码: " +
                                     std::to_string(static_cast<int>(st)));
          return result;
        },
        nb::arg("x"), nb::sig("def argmax(x) -> Tensor"));

  // ---- 信息 ----
  m.def("cuda_available", []() {
#ifdef AXONO_WITH_CUDA
    return true;
#else
    return false;
#endif
  });

#ifdef AXONO_WITH_CUDA
  // ---- cuBLASLt 开关 ----
  m.def("use_cublas_lt", [](bool enable) {
         ops::cuda::UseCublasLt(enable);
       }, nb::arg("enable"), nb::sig("def use_cublas_lt(enable: bool) -> None"));
  m.def("cublas_lt_enabled", []() {
         return ops::cuda::CublasLtEnabled();
       });

  // ---- CUDA Graph ----
  // 低层流句柄 API (供 Python 上下文管理器使用)
  m.def("_graph_begin_capture", []() {
         // 预热 matmul 后端句柄/workspace: cuBLAS/Lt 首次调用会做内部分配,
         // 若发生在捕获体内会被判非法 (捕获禁止分配/同步)。
         // 用 1x1 matmul 走一遍完整路径 (含 Lt heuristic + 回退), 仅一次。
         static std::once_flag warm_once;
         std::call_once(warm_once, [] {
           // 用带设备串的构造建 CUDA 张量 (Create() 是 CPU 的)
           core::Tensor a(core::DataType::FLOAT32, core::Shape{8, 8},
                          "cuda");
           core::Tensor b(core::DataType::FLOAT32, core::Shape{8, 8},
                          "cuda");
           core::Tensor c(core::DataType::FLOAT32, core::Shape{8, 8},
                          "cuda");
           ops::cuda::MatMul(core::Context(), a, b, c);
         });
         // 捕获流是 non-blocking, 与默认流无隐式同步 —— 捕获前必须等
         // 所有已提交的 kernel (如输入填充) 完成, 否则图内 kernel 与
         // 它们产生竞态 (回放读到未写入的数据)。
         cudaDeviceSynchronize();
         cudaStream_t stream = nullptr;
         core::Status st = core::cuda::BeginGraphCapture(&stream);
         if (st != core::Status::OK)
           throw std::runtime_error("_graph_begin_capture 失败");
         return reinterpret_cast<intptr_t>(stream);
       });
  m.def("_graph_end_capture",
        [](intptr_t handle, core::cuda::CudaGraphExec &exec) {
          auto stream = reinterpret_cast<cudaStream_t>(handle);
          core::Status st = core::cuda::EndGraphCapture(stream, &exec);
          if (st != core::Status::OK)
            throw std::runtime_error(
                "_graph_end_capture: 捕获体内含非法调用 (同步/H2D/分配) "
                "或未提交任何 kernel");
        });
  m.def("_graph_abort_capture", [](intptr_t handle) {
         core::cuda::AbortGraphCapture(
             reinterpret_cast<cudaStream_t>(handle));
       });

  // 用法: g = CUDAGraph(); g.capture(fn); g.replay(); g.reset()
  // capture 体内只能提交 CUDA 算子 (matmul/add/relu), 不能有 H2D/D2H。
  nb::class_<core::cuda::CudaGraphExec>(m, "CUDAGraph")
      .def(nb::init<>())
      .def(
          "capture",
          [](core::cuda::CudaGraphExec &self, nb::callable fn) {
            // 1) warmup: 登记模式跑一遍, 所有 CUDA 分配进入捕获池
            core::cuda::ResetCapturePool();
            core::cuda::SetCapturePoolRecording(true);
            try {
              fn();
            } catch (...) {
              core::cuda::SetCapturePoolRecording(false);
              throw;
            }
            core::cuda::SetCapturePoolRecording(false);

            // 2) 正式捕获: 同尺寸分配命中池 (地址不变), 图内仅 kernel 节点
            cudaStream_t stream = nullptr;
            core::Status st = core::cuda::BeginGraphCapture(&stream);
            if (st != core::Status::OK)
              throw std::runtime_error("CUDAGraph.capture: 开始捕获失败");
            try {
              fn();  // 体内算子经 AxonoCurrentStream() 提交到捕获流
            } catch (...) {
              core::cuda::AbortGraphCapture(stream);
              throw;
            }
            st = core::cuda::EndGraphCapture(stream, &self);
            if (st != core::Status::OK)
              throw std::runtime_error(
                  "CUDAGraph.capture: 捕获体内含有非法调用 "
                  "(同步/内存分配/H2D拷贝) 或未提交任何 kernel");
          },
          nb::arg("fn"), nb::sig("def capture(self, fn) -> None"))
      .def(
          "replay",
          [](core::cuda::CudaGraphExec &self) {
            core::Status st = self.Replay(0);
            if (st != core::Status::OK)
              throw std::runtime_error(
                  "CUDAGraph.replay: 图未捕获或回放失败");
          },
          nb::sig("def replay(self) -> None"))
      .def(
          "sync",
          [](core::cuda::CudaGraphExec &self) {
            core::Status st = self.Sync();
            if (st != core::Status::OK)
              throw std::runtime_error("CUDAGraph.sync 失败");
          },
          nb::sig("def sync(self) -> None"))
      .def(
          "reset",
          [](core::cuda::CudaGraphExec &self) {
            // 图与捕获池一并回收 (池里的块专供本次捕获复用)
            self.Reset();
            core::cuda::ResetCapturePool();
          },
          nb::sig("def reset(self) -> None"))
      .def_prop_ro("is_captured", &core::cuda::CudaGraphExec::IsCaptured)
      .def_prop_ro("num_nodes", &core::cuda::CudaGraphExec::NumNodes);
#endif

  m.attr("__version__") = "0.2.0";
}
