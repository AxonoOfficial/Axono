# Axono 顶层 CMake —— v0.2 (nanobind 框架)
# 用法:
#   CPU:  cmake -B build -DPython_EXECUTABLE=$(which python3)
#   CUDA: cmake -B build -DWITH_CUDA=ON -DPython_EXECUTABLE=$(which python3)
cmake_minimum_required(VERSION 3.18)

# CUDA 必须在 project() 之前决定语言
if(WITH_CUDA)
  project(axono LANGUAGES CXX CUDA)
else()
  project(axono LANGUAGES CXX)
endif()

# where python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# POLICY
if(POLICY CMP0007)
    cmake_policy(SET CMP0007 NEW)
endif()

# INFO 展示
function(showInfo text)
    if(CMAKE_HOST_UNIX)
        message("-- ${text}")
    else()
        message(STATUS "-- ${text}")
    endif()
endfunction()

# CMake 基本设定档
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/wizard/"
)

# C++ / CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -fPIC)
    add_link_options(-fPIC)
endif()

# OpenMP: CPU elementwise/matmul 内核并行
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
    add_link_options(${OpenMP_CXX_FLAGS})
    message(STATUS "Axono: OpenMP enabled -> ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "Axono: OpenMP not found, CPU kernels run single-threaded")
endif()

# x86 架构检测 (AVX2 优化)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i386|i686|AMD64|amd64")
    message(STATUS "Detected x86 architecture, enabling AVX optimizations")
    add_compile_definitions(AXONO_USE_X86_INTRINSICS=1)
    include(CheckCXXCompilerFlag)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            add_compile_options(-mavx2 -mfma)
        endif()
    endif()
else()
    message(STATUS "Non-x86 architecture, using generic implementation")
    add_compile_definitions(AXONO_USE_X86_INTRINSICS=0)
endif()

# 对于 Nvidia 用户
option(WITH_CUDA "编译 Axono 时附带 CUDA 支持" OFF)
option(WITH_CINN "编译 Axono 时附带 CINN 支持 (预留)" OFF)
option(WITH_NCCL "编译 Axono 时附带 NCCL 支持 (预留)" OFF)

if(WITH_CUDA)
  showInfo("CUDA Support - ON")
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
else()
  showInfo("CUDA Support - OFF")
endif()
if(WITH_CINN)
  showInfo("CINN Support - ON (预留)")
endif()
if(WITH_NCCL)
  showInfo("NCCL Support - ON (预留)")
endif()

# 第三方 THIRD PARTY (nanobind 等)
include(third_party)

# 核心编译
add_subdirectory(axono)
