// Axono v0.2 — nanobind 绑定模块
// 取代 dev 分支的 pybind11_module.cpp + include/axono/pybind/*。
// 模块名保持 libaxono, Python 端 from libaxono import ... 不变。

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
#include "axono/core/cuda/tensor/kernel.h"
#include "axono/ops/cuda/add.h"
#include "axono/ops/cuda/matmul.h"
#include "axono/ops/cuda/relu.h"
#endif
#include "axono/ops/cpu/add.h"
#include "axono/ops/cpu/matmul.h"
#include "axono/ops/cpu/relu.h"

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

  m.def("relu", [](const core::Tensor &input, bool inplace) {
    if (inplace) {
      core::Tensor t = input;  // 拷贝后原地
      core::Status st;
      if (t.is_cuda()) {
#ifdef AXONO_WITH_CUDA
        st = ops::cuda::ReluInplace(core::Context(), t);
#else
        st = core::Status::DEVICE_ERROR;
#endif
      } else {
        st = ops::cpu::ReluInplace(core::Context(), t);
      }
      if (st != core::Status::OK)
        throw std::runtime_error("relu(inplace) 失败, 错误代码: " +
                                 std::to_string(static_cast<int>(st)));
      return t;
    }
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
  }, nb::arg("x"), nb::arg("inplace") = false);

  // ---- 信息 ----
  m.def("cuda_available", []() {
#ifdef AXONO_WITH_CUDA
    return true;
#else
    return false;
#endif
  });
  m.attr("__version__") = "0.2.0";
}
