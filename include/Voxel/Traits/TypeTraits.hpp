//==--- Voxel/Traits/TypeTraits.hpp ------------------------ -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  TypeTraits.hpp
/// \brief This file defines general traits.
// 
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Container/Tuple.hpp>
#include <type_traits>

namespace Voxx   {
namespace Traits {

/// Returns true if the type is an lvalue reference and it is const.
/// \tparam T The type to check for const lvalue reference.
template <typename T>
struct is_const_lvalue_reference {
  /// Returns true if T is a constant lvalue reference.
  static constexpr bool value =
    std::is_lvalue_reference<T>::value &&
    std::is_const<std::remove_reference_t<T>>::value;
};

/// Returns true if the type is an lvalue reference and it is not const.
/// \tparam T The type to check for const lvalue reference.
template <typename T>
struct is_non_const_lvalue_reference {
  /// Returns true if T is a constant lvalue reference.
  static constexpr bool value = 
    std::is_lvalue_reference<T>::value &&
    !std::is_const<std::remove_reference_t<T>>::value;
};

template <typename... Ts>
struct has_const_lvalue_reference {
  static constexpr bool value = 0;
};

/// Returns true if one of the parameter pack containers a const lvalue
/// reference.
/// \param  Type    The type of the first type to check.
/// \tparam Types   The types to check for const lvalue reference.
template <typename Type, typename... Types>
struct has_const_lvalue_reference<Type, Types...> {
  /// Returns true of one of the types is a const lvalue reference.
  static constexpr bool value =
    is_const_lvalue_reference<Type>::value +
    has_const_lvalue_reference<Types...>::value;
};

template <typename... Ts>
struct has_non_const_lvalue_reference {
  static constexpr bool value = 0;
};

/// Returns true if one of the parameter pack containers an lvalue reference
/// which is not const.
/// \tparam Types   The types to check for non const lvalue reference.
template <typename Type, typename... Types>
struct has_non_const_lvalue_reference<Type, Types...> {
  /// Returns true of one of the types is a non const lvalue reference.
  static constexpr bool value =
    is_non_const_lvalue_reference<Type>::value +
    has_non_const_lvalue_reference<Types...>::value;
};

//==--- _v and _t versions -------------------------------------------------==//

/// Returns true if T is both lvalue reference and const.
/// \tparam T The type to check for const lvalue reference.
template <typename T> static constexpr bool
is_const_lvalue_reference_v = is_const_lvalue_reference<T>::value;

/// Returns true if T is lvalue reference and not const.
/// \tparam T The type to check for const lvalue reference.
template <typename T> static constexpr bool
is_non_const_lvalue_reference_v = is_non_const_lvalue_reference<T>::value;

/// Returns true if one of the parameter pack containers an lvalue reference
/// which is const.
/// \tparam Ts   The types to check for const lvalue reference.
template <typename... Ts> static constexpr bool
has_non_const_lvalue_reference_v = has_non_const_lvalue_reference<Ts...>::value;

/// Returns true if one of the parameter pack containers an lvalue reference
/// which is not const.
/// \tparam Ts   The types to check for non const lvalue reference.
template <typename... Ts> static constexpr bool
has_const_lvalue_reference_v = has_const_lvalue_reference<Ts...>::value;

}} // namespace Voxx:Traits