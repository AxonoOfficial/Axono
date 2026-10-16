// axono/core/ops.h — 算子注册表 (v0.2, 无 pybind 依赖)
#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "axono/core/tensor.h"

namespace axono {
namespace core {

// 算子函数签名: 接收输入 Tensor 列表, 返回输出 Tensor。
// 算子以名字字符串注册, Python 侧通过 libaxono 绑定按名调用。
using OpFunction = std::function<Tensor(const std::vector<Tensor> &)>;

// 全局算子注册表 (Meyers 单例): 名字 -> OpFunction。
// register_op 注册 (通常由各算子文件的静态注册器完成),
// run/op_names 供调试与动态分发使用。
class OpRegistry {
 public:
  static OpRegistry &instance() {
    static OpRegistry registry;
    return registry;
  }

  void register_op(const std::string &name, OpFunction func) {
    ops_[name] = std::move(func);
  }

  bool has_op(const std::string &name) const {
    return ops_.find(name) != ops_.end();
  }

  const OpFunction &get_op(const std::string &name) const {
    auto it = ops_.find(name);
    if (it == ops_.end()) {
      throw std::runtime_error("op not registered: " + name);
    }
    return it->second;
  }

  // 便捷调用
  Tensor run(const std::string &name, const std::vector<Tensor> &inputs) const {
    return get_op(name)(inputs);
  }

  std::vector<std::string> op_names() const {
    std::vector<std::string> names;
    names.reserve(ops_.size());
    for (const auto &kv : ops_) names.push_back(kv.first);
    return names;
  }

 private:
  OpRegistry() = default;
  std::unordered_map<std::string, OpFunction> ops_;
};

}  // namespace core
}  // namespace axono
