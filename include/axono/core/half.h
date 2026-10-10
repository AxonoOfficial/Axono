#pragma once

// IEEE 754 binary16 <-> binary32 转换 (位操作实现, 无外部依赖)。
// 参考 Fabian Giesen 的 public-domain half 实现精简而来。

#include <cstdint>

namespace axono {
namespace core {
namespace detail {

// float32 -> float16 (返回 IEEE half 位模式)。溢出饱和到 inf, 次正规舍入。
inline uint16_t Float2Half(float value) {
  uint32_t x;
  __builtin_memcpy(&x, &value, sizeof(x));

  const uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t exp = (x >> 23) & 0xFFu;
  uint32_t man = x & 0x7FFFFFu;

  if (exp == 0xFF) {  // inf / nan
    return static_cast<uint16_t>(sign | 0x7C00u |
                                 (man ? (man >> 13 ? 0x0200u : 1) : 0));
  }
  // 重新偏置: fp32 exp-127 -> fp16 exp-15
  int32_t e = static_cast<int32_t>(exp) - 127 + 15;
  if (e >= 0x1F) {  // 溢出 -> inf
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  if (e <= 0) {  // 下溢 -> 次正规或零
    if (e < -10) return static_cast<uint16_t>(sign);
    man |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - e);
    const uint32_t round = (1u << (shift - 1));
    uint16_t half_man =
        static_cast<uint16_t>((man >> shift) + ((man & round) != 0 &&
                                                ((man >> shift) & 1 ||
                                                 (man & (round << 1)) != 0)
                                                    ? 1
                                                    : 0));
    return static_cast<uint16_t>(sign | half_man);
  }
  // 舍入到最近偶数
  const uint32_t round = 0x00001000u;
  uint32_t m = man + (((man & round) != 0 && (man & (round << 1)) != 0) ? 0x1u
                                                                        : 0x0u);
  // 舍入进位可能溢出尾数
  uint32_t half = (static_cast<uint32_t>(e) << 10) | (m >> 13);
  return static_cast<uint16_t>(sign | half);
}

// float16 (IEEE half 位模式) -> float32
inline float Half2Float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  uint32_t x;
  if (exp == 0) {
    if (man == 0) {  // 零
      x = sign;
    } else {  // 次正规 -> 正规化
      int32_t e = -1;
      uint32_t m = man;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFu;
      x = sign | (static_cast<uint32_t>(127 - 15 + e + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1F) {  // inf / nan
    x = sign | 0x7F800000u | (man << 13);
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out;
  __builtin_memcpy(&out, &x, sizeof(out));
  return out;
}

}  // namespace detail
}  // namespace core
}  // namespace axono
