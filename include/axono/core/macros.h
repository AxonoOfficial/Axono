#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define AXONO_FORCE_INLINE inline __attribute__((always_inline))
#define AXONO_LIKELY(x) __builtin_expect(!!(x), 1)
#define AXONO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define AXONO_FORCE_INLINE inline
#define AXONO_LIKELY(x) (x)
#define AXONO_UNLIKELY(x) (x)
#endif

#define AXONO_EXPORT
#define AXONO_API

#define AXONO_DISALLOW_COPY(ClassName)   \
  ClassName(const ClassName &) = delete; \
  ClassName &operator=(const ClassName &) = delete

#define AXONO_DISALLOW_MOVE(ClassName) \
  ClassName(ClassName &&) = delete;    \
  ClassName &operator=(ClassName &&) = delete

#define AXONO_DISALLOW_COPY_AND_MOVE(ClassName) \
  AXONO_DISALLOW_COPY(ClassName);               \
  AXONO_DISALLOW_MOVE(ClassName)

#if defined(__CUDACC__) || defined(__NVCC__)
#define AXONO_COMPILING_WITH_NVCC 1
#else
#define AXONO_COMPILING_WITH_NVCC 0
#endif
