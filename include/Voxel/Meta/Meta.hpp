//==--- Voxel/Meta/Meta.hpp -------------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  Meta.hpp
/// \brief This file defines meta programming related functionality.
//
//==------------------------------------------------------------------------==//

#ifndef VOXX_META_META_HPP
#define VOXX_META_META_HPP

namespace Voxx   {
namespace Detail {

/// The PackByteSize struct computes the number of bytes in a parameter pack.
/// \tparam   Ts  The types in the paramter pack.
template <typename... Ts> struct PackByteSize {};

/// Specialization of the PackByteSize struct for the recursive case.
/// \tparam   T   The type whose size is added to the total pack size.
/// \tparam   Ts  The rest of the types whose size must still be added.
template <typename T, typename... Ts>
struct PackByteSize<T, Ts...> {
  /// Returns the size (in bytes) of the parameter pack.
  static constexpr std::size_t size = sizeof(T) + PackByteSize<Ts...>::size;
};

/// Specialization of the PackBytes struct for the termination case.
template <>
struct PackByteSize<> {
  /// Returns that the size of add is zero.
  static constexpr std::size_t size = 0;
};

} // namespace Detail

/// Returns the number of bytes which the paramter pack will require.
/// \tparam   Pack  The pack to determine the number of bytes for.
template <typename... Pack>
static constexpr std::size_t packByteSize = Detail::PackByteSize<Pack...>::size;

} // namespace Voxx::Detail

#endif // VOXX_META_META_HPP