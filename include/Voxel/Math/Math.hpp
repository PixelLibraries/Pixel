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

#include <type_traits>

namespace Voxx {
namespace Math {

/// Returns true if the parameter is a power of 2, for any integral type. For
/// \p value = 0, this will return false.
/// \param[in]  value   The value to check if is a power of 2.
/// \tparam     T       The type of the value.
template <typename T>
static constexpr bool isPowerOfTwo(T&& value) noexcept {
  static_assert(std::is_integral<std::decay_t<T>>::value, "Non integral type");
  return !(value == 0) && !(value & (value - 1));
}

}} // namespace Voxx::Math