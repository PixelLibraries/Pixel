//==--- Voxel/Container/Detail/TupleImpl.hpp --------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  TupleImpl.hpp
/// \brief This file defines the implementation of the core Tuple functionality.
// 
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Utility/Portability.hpp>
#include <utility>

namespace Voxx   {
namespace Detail {

//==--- Element -----------------------------------------------------------==//

/// Defines a struct to hold an element at a specific index in a container.
/// \tparam Index The index of the element being held.
/// \tparam Type  The type of the element.
template <size_t Index, typename Type>
struct Element {
  /// Default constructor, which does nothing.
  VoxxDeviceHost constexpr Element() noexcept = default;

  /// Constructor -- Sets the element, casting it to the type held by the
  /// Element.
  /// \param  element     The element to use to set this Element's data.
  /// \tparam ElementType The type of the element.
  template <typename ElementType>
  VoxxDeviceHost explicit constexpr Element(ElementType&& element) noexcept
  : value(std::forward<Type>(element)) {}

  Type value;  //!< The element being held.
};

//==--- getImpl -----------------------------------------------------------==//

/// Gets a constant lvalue-reference to an Element.
/// \param[in] element The element to get.
/// \tparam    Index   The index of the element to get.
/// \tparam    Type    The type of the element to get.
template <size_t Index, typename Type> 
VoxxDeviceHost constexpr inline const Type&
getImpl(const Element<Index, Type>& element) noexcept {
  return element.value;
}

/// Gets an lvalue-reference to an Element.
/// \param[in] element The element to get.
/// \tparam    Index   The index of the element to get.
/// \tparam    Type    The type of the element to get.
template <size_t Index, typename Type> 
VoxxDeviceHost constexpr inline Type&
getImpl(Element<Index, Type>& element) noexcept {
  return element.value;
}

/// Gets an rvalue-reference to an Element.
/// \param[in] element The element to get.
/// \tparam    Index   The index of the element to get.
/// \tparam    Type    The type of the element to get.
template <size_t Index, typename Type> 
VoxxDeviceHost constexpr inline Type&&
getImpl(Element<Index, Type>&& element) noexcept {
  return std::move(element.value);
}

/// Gets a constant lvalue-reference to an Element.
/// \param[in] element The element to get.
/// \tparam    Index   The index of the element to get.
/// \tparam    Type    The type of the element to get.
template <size_t Index, typename Type> 
VoxxDeviceHost constexpr inline const volatile Type&
getImpl(const volatile Element<Index, Type>& element) noexcept {
  return element.value;
}

/// Gets an lvalue-reference to an Element.
/// \param[in] element The element to get.
/// \tparam    Index   The index of the element to get.
/// \tparam    Type    The type of the element to get.
template <size_t Index, typename Type> 
VoxxDeviceHost constexpr inline volatile Type&
getImpl(volatile Element<Index, Type>& element) noexcept {
  return element.value;
}

/// Gets an rvalue-reference to an Element.
/// \param[in] element The element to get.
/// \tparam    Index   The index of the element to get.
/// \tparam    Type    The type of the element to get.
template <size_t Index, typename Type> 
VoxxDeviceHost constexpr inline volatile Type&&
getImpl(volatile Element<Index, Type>&& element) noexcept {
  return std::move(element.value);
}

//==--- TupleStorage ------------------------------------------------------==//

/// Defines the implementation of the storage for Tuple types. The elements
/// which are stored are laid out in memory in the order in which they are
/// present in the parameter pack. For example:
///
/// \code{.cpp}
///   TupleStorage<Indices, float, int, double> ...
///
///   // Laid out as follows:
///   float   // 4 bytes
///   int     // 4 bytes
///   double  // 8 bytes
/// \endcode
///
/// \tparam Indices The indices of the tuple elements.
/// \tparam Types   The types of the tuple elements.
template <typename Indicies, typename... Types>
struct TupleStorage;

/// Specialization for the implementation of the TupleStorage.
/// \tparam Indices The indices for the locations of the elements.
/// \tparam Types   The types of the elements.
template <size_t... Indices, typename... Types>
struct TupleStorage<std::index_sequence<Indices...>, Types...> 
: Element<Indices, Types>... {
  /// Defines the size (number of elements) of the tuple storage.
  static constexpr size_t elements = sizeof...(Types);

  /// Default constructor -- default implementation.
  VoxxDeviceHost constexpr TupleStorage() noexcept = default;

  /// Constructor -- Sets the elements of the tuple. This overload is selected 
  /// when the element are forwarding reference types. The types are forwarded
  /// as the tuple types (i.e Types) rather than as the deduced types for the
  /// parameters for this function (i,e ElementTypes).
  /// 
  /// \param  elements     The elements to use to set the tuple.
  /// \tparam ElementTypes The types of the elements.  
  template <typename... ElementTypes>
  VoxxDeviceHost constexpr TupleStorage(ElementTypes&&... elements) noexcept
  : Element<Indices, Types>(std::forward<Types>(elements))...
  {}

  /// Constructor -- Sets the elements of the tuple. This overload is selected 
  /// when the elements are const lvalue references, in which case they need to
  /// be copied. The types are forwarded as the tuple types (i.e Types) rather
  /// than as the deduced types for the parameters for this function
  /// (i.e ElementTypes).
  ///  
  /// \param  elements     The elements to use to set the tuple.
  /// \tparam ElementTypes The types of the elements.  
  template <typename... ElementTypes>
  VoxxDeviceHost TupleStorage(const ElementTypes&... elements) noexcept
  : Element<Indices, Types>(std::forward<Types>(elements))...
  {}
};

//==--- BasicTuple --------------------------------------------------------==//

/// Defines a basic tuple class, which essentially is just a cleaner interface 
/// for TupleStorage. The Types are laid out in memory in the same order in
/// which they appear in the parameter pack. For example:
///
/// ~~~cpp
///   BasicTuple<int, float, double> tuple(4, 3.14f, 2.7);
///
///   // Laid out as follows:
///   int     : 4      // 4 bytes
///   float   : 3.14   // 4 bytes
///   double  : 2.7    // 8 bytes
/// ~~~
///
/// \tparam Types The types of the tuple elements.
template <typename... Types>
struct BasicTuple :
TupleStorage<std::make_index_sequence<sizeof...(Types)>, Types...> {
 public:
  //==--- Alises -----------------------------------------------------------==//

  /// Alias for the index sequence.
  using IndexSequence = std::make_index_sequence<sizeof...(Types)>;
  /// Alias for the base type of the Tuple.
  using BaseType      = TupleStorage<IndexSequence, Types...>;

  //==--- Constants --------------------------------------------------------==//
  
  /// Defines the size (number of elements) of the BasicTuple.
  static constexpr size_t elements = BaseType::elements;

 public:
  /// Default constructor -- implements default behaviour.
  VoxxDeviceHost constexpr BasicTuple() noexcept = default;

  /// Creates a BasicTuple from a variadic list of elements. This overload is
  /// called if the the \p elements are forwarding reference types.
  /// \param[in] elements     The elements for the BasicTuple.
  /// \tparam    ElementTypes The types of the elements for the BasicTuple.
  template <typename... ElementTypes>
  VoxxDeviceHost explicit constexpr
  BasicTuple(ElementTypes&&... elements) noexcept 
  : BaseType(std::forward<Types>(elements)...) {}

  /// Creates a BasicTuple from a variadic list of elements. This overload is
  /// called if the the \p elements are constant lvalue reference types.
  /// \param[in] elements     The elements for the BasicTuple.
  /// \tparam    ElementTypes The types of the elements for the BasicTuple.  
  template <typename... ElementTypes>
  VoxxDeviceHost constexpr explicit
  BasicTuple(const ElementTypes&... elements) noexcept
  : BaseType(std::forward<Types>(elements)...) {}
};  

}} // namespace Voxx::Detail
