//==--- Voxel/Utility/Bitwise.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  Bitwise.hpp
/// \brief This file defines bit related functionality.
//
//==------------------------------------------------------------------------==//

#pragma once

#include <cstdlib>

namespace Voxx {

/// Computes the value with only the least significant bit set.
/// \param[in]  value  The value to get the least significant bit of.
/// \tparam     T      The type of the value.
/// Returns the value with only the first non zero bit set.
template <typename T>
static constexpr auto leastSetBitOnly(T value) noexcept -> T {
  return value & ~(value - 1);
}

/// Gets the index of the first set bit, by calculating the log of the value
/// with only the least non zero bit set. The implementation is taken from:
/// 
///   https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
///   
/// \param[in]  val   The value to get the index of the least set bit of.
/// Returns the value of the index of the first set bit.
static constexpr auto firstSetBitIndex(uint64_t val) noexcept -> uint64_t {
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