#pragma once

#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace core {
namespace cuda {
namespace tensor {

AXONO_EXPORT Status TransposeKernel(const Tensor& src, Tensor& dst, int dim0, int dim1);

}  // namespace tensor
} // namespace cuda
}  // namespace core
}  // namespace axono
