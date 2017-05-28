//==--- Voxel/Math/Math.hpp -------------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  Math.hpp
/// \brief This file defines math related functionality.
// 
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Utility/Portability.hpp>
#include <type_traits>

namespace Voxx   {
namespace Math   {
namespace Detail {

/// Seed for the random number generator.
static uint32_t randSeed = 123456789;

/// Implementation of a random number generator using the xor32 method, as
/// described here:
/// 
///   [xorshift32](https://en.wikipedia.org/wiki/Xorshift)
/// 
/// Returns a random 32 bit integer in the range [0, 2^32 - 1].
VoxxDeviceHost inline uint32_t xorshift32() noexcept {
  uint32_t x = randSeed;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  randSeed = x;
  return x; 
}

} // namespace Detail

/// Returns true if the parameter is a power of 2, for any integral type. For
/// \p value = 0, this will return false.
/// \param[in]  value   The value to check if is a power of 2.
/// \tparam     T       The type of the value.
template <typename T>
VoxxDeviceHost constexpr inline bool isPowerOfTwo(T&& value) noexcept {
  static_assert(std::is_integral<std::decay_t<T>>::value, "Non integral type");
  return !(value == 0) && !(value & (value - 1));
}

/// Computes the log_2 of a 32 bit integer.
/// \param[in] value The value to find the log_2 of.
VoxxDeviceHost constexpr inline uint32_t log2(uint32_t value) noexcept {
  uint32_t result = 0, shift = 0;
  result = (value > 0xFFFF) << 4; value >>= result;
  shift  = (value > 0xFF  ) << 3; value >>= shift ; result |= shift;
  shift  = (value > 0xF   ) << 2; value >>= shift ; result |= shift;
  shift  = (value > 0x3   ) << 1; value >>= shift ; result |= shift;
  return result |= (value >> 1);
}

/// Returns a random 32 bit integer in the range [\p start. \p end]. It's
/// extremely fast (benchmarks show approx ~1-2ns, in comparison 
/// std::experimental::randint is around 16ns), however, the implementation
/// is known to fail some of the random number generation tests, and also
/// only provides a uniform distribution. It should be used only when
/// performance is critical and the quality of the generated random number is
/// not hugely important.
/// 
/// \param[in]  start   The starting value of the range.
/// \param[in]  end     The end value of the range.
/// \param[in]  seed    The seed to use for the generator.
/// Returns a uniformly distributed random number between 
VoxxDeviceHost inline uint32_t randint(uint32_t start, uint32_t end) noexcept {
  const uint32_t range = end - start;
  return (Detail::xorshift32() >> (32 - log2(range) - 1)) % range + start;
}

}} // namespace Voxx::Math