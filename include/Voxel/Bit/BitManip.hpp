//==--- Voxel/Bit/BitManip.hpp ----------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  BitManip.hpp
/// \brief This file defines bit manipulation functionality.
//
//==------------------------------------------------------------------------==//

#pragma once

#include <type_traits>
#include <iostream>

namespace Voxx {

/// Sets the bit at position \p n in \p data to the value \p value.
/// \tparam T The type of the data.
template <typename T>
constexpr inline void setBit(T& data, uint8_t n, bool value) noexcept {
  data ^= (-value ^ data) & (1ul << n);
}

/// Gets the bit at position \p n in \p data.
/// \tparam T The type of the data.
template <typename T>
constexpr inline bool getBit(T data, uint8_t n) noexcept {
  return (data >> n) & 0x01 ;
}

/// Gets the bits between \p start and \p end, inclusive, and returns the
/// result.
/// \tparam T The type of the data.
template <typename T>
constexpr inline T getBits(T data, uint8_t start, uint8_t end) noexcept {
//  return (data >> start) & ((1 << end) - 1); 
  return (data >> start) & ((1ul << (end - start + 1)) - 1); 
}

/// Computes the value with only the least significant bit set.
/// \param[in]  value  The value to get the least significant bit of.
/// \tparam     T      The type of the value.
/// Returns the value with only the first non zero bit set.
template <typename T>
constexpr inline std::decay_t<T> leastSetBitOnly(T value) noexcept {
  return value & ~(value - 1);
}

/// Gets the index of the first set bit, by calculating the log of the value
/// with only the least non zero bit set. The implementation is taken from:
/// 
///   https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
///   
/// \param[in]  val   The value to get the index of the least set bit of.
/// Returns the value of the index of the first set bit.
constexpr inline uint64_t firstSetBitIndex(uint64_t val) noexcept {
  constexpr uint64_t shift[] = {1, 2, 4, 8, 16, 32};
  constexpr uint64_t mask[]  = {0x0000000000000002,
                                0x000000000000000C,
                                0x00000000000000F0,
                                0x000000000000FF00,
                                0x00000000FFFF0000,
                                0xFFFFFFFF00000000};

  uint64_t result = 0, value = leastSetBitOnly(val);
  for (int i = 5; i >= 0; i--) {
    if (value & mask[i]) {
      value  >>= shift[i];
      result  |= shift[i];
    }
  }
  return result;
}


} // namespace Voxx