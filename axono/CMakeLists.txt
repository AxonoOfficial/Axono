# Axono 核心编译 —— v0.2 (nanobind 框架)
# 模块名保持 libaxono，与 Python 端 from libaxono import ... 兼容
# 源码布局:
#   ../src      -> C++ 核心 (CPU + 通用)
#   ../src/ops/cuda, ../src/core/cuda -> CUDA 源 (仅 WITH_CUDA)
#   python/src  -> nanobind 绑定

# 收集源码
file(GLOB_RECURSE AXONO_SRCS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp")

set(AXONO_CUDA_SRCS "")
if(WITH_CUDA)
  file(GLOB_RECURSE AXONO_CUDA_SRCS CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu")
endif()

list(LENGTH AXONO_SRCS AXONO_SRCS_COUNT)
list(LENGTH AXONO_CUDA_SRCS AXONO_CUDA_SRCS_COUNT)
showInfo("Axono core: ${AXONO_SRCS_COUNT} C++ files, ${AXONO_CUDA_SRCS_COUNT} CUDA files")

nanobind_add_module(libaxono
    NB_STATIC
    STABLE_ABI
    LTO
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/nanobind_module.cpp
    ${AXONO_SRCS}
    ${AXONO_CUDA_SRCS}
)

target_include_directories(libaxono PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

if(WITH_CUDA)
  target_compile_definitions(libaxono PRIVATE WITH_CUDA AXONO_WITH_CUDA)
  # cuBLAS
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(libaxono PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(libaxono PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON)
endif()

# CPU BLAS 加速 (OpenBLAS 优先): 找到则定义 AXONO_WITH_BLAS,
# CPU matmul float/double 走 cblas_sgemm/cblas_dgemm, 自写 kernel 作回退。
set(AXONO_BLAS_IMPL "none")
find_package(BLAS QUIET)
if(BLAS_FOUND)
  target_link_libraries(libaxono PRIVATE ${BLAS_LIBRARIES})
  target_compile_definitions(libaxono PRIVATE AXONO_WITH_BLAS)
  set(AXONO_BLAS_IMPL "generic-BLAS")
  message(STATUS "Axono: BLAS found -> ${BLAS_LIBRARIES}")
else()
  message(STATUS "Axono: BLAS not found, using built-in CPU kernels")
endif()
# OpenBLAS 专属: 多线程与更优内核 (cblas 接口)
find_library(OPENBLAS_LIB NAMES openblas HINTS ${BLAS_LIB_DIR})
if(OPENBLAS_LIB)
  target_link_libraries(libaxono PRIVATE ${OPENBLAS_LIB})
  target_compile_definitions(libaxono PRIVATE AXONO_WITH_OPENBLAS)
  set(AXONO_BLAS_IMPL "OpenBLAS")
  message(STATUS "Axono: OpenBLAS -> ${OPENBLAS_LIB}")
endif()
showInfo("Axono CPU BLAS: ${AXONO_BLAS_IMPL}")

# 安装到 python 包内
install(TARGETS libaxono LIBRARY DESTINATION axono)
