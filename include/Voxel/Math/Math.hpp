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

/// Implementation of a random number generator using the xor32 method, as
/// described here:
/// 
///   [xorshift32](https://en.wikipedia.org/wiki/Xorshift)
/// 
/// \param[in]  seed  The seed to use for the generator.
/// Returns a random 32 bit integer in the range (0, 2^32 - 1).
VoxxDeviceHost constexpr inline uint32_t xorshift32(uint32_t seed) noexcept {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed; 
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

/// Returns a random 32 bit integer in the range [\p start. \p end]. It's
/// extremely fast (benchmarks show approx ~2ns), however, the implementation
/// is known to fail some of the random number generation tests, and also
/// only provides a uniform distribution. It should be used only when
/// performance is critical and the quality of the generated random number is
/// not hugely important.
/// \param[in]  start   The starting value of the range.
/// \param[in]  end     The end value of the range.
/// \param[in]  seed    The seed to use for the generator.
/// Returns a uniformly distributed random number between 
VoxxDeviceHost constexpr inline uint32_t
randint(uint32_t start, uint32_t end, uint32_t seed = 123456789) noexcept {
  return Detail::xorshift32(seed) % (end - start) + start;
}


}} // namespace Voxx::Math