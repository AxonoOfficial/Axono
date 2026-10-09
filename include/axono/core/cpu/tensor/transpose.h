#pragma once

#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace core {
namespace cpu {
namespace tensor {

AXONO_EXPORT Status TransposeKernel(const Tensor& src, Tensor& dst, int dim0, int dim1);

}  // namespace tensor
} // namespace cpu
}  // namespace core
}  // namespace axono
