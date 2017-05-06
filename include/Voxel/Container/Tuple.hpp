//==--- Voxel/Container/Tuple.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  Tuple.hpp
/// \brief This file defines a host and device Tuple class.
// 
//==------------------------------------------------------------------------==//

#pragma once

#include "Detail/TupleImpl.hpp"

namespace Voxx {

//==--- Tuple -------------------------------------------------------------==//

/// Defines a struct to hold a fixed-size number of heterogenous types. This 
/// implementation of Tuple is similar to std::tuple, but allows the tuple
/// to be used on both the host and the device. We could use thrust::tuple,
/// however, it only supports 9 types (rather than the compiler's recursion
/// depth limit in out case), since it doesn't use variadic tempates, which is
/// also limiting.
/// 
/// The interface through which a tuple should be created is the make_tuple()
/// function, i,e:
/// 
/// ~~~cpp
/// auto tuple = make_tuple(4, 3.5, "some value");  
/// ~~~
/// 
/// unless the tuple needs to store reference types. See the documentation for
/// make_tuple for an example.
/// 
/// \tparam Types The types of the elements in the tuple.
template <typename... Types>
struct Tuple;

/// Specialization for an empty Tuple.
template <>
struct Tuple<> {
  /// Intializes the Tuple with no elements.
  VoxxDeviceHost constexpr Tuple() noexcept {}

  /// Alias for the storage type.
  using StorageType = Detail::BasicTuple<>;

  /// Defines the size of the Tuple.
  static constexpr size_t elements = StorageType::elements;
};

/// Specialization for a non-empty Tuple.
/// \tparam Types The types of the Tuple elements.
template <typename... Types> 
struct Tuple {
 private:
  /// Alias for the type of the Tuple.
  using SelfType = Tuple<Types...>;
  /// Alias for std::is_same_v which can be replaced for c++17.
  template <typename T, typename U>
  static constexpr bool is_same_v = std::is_same<T, U>::value;

  /// Returns true of the FirstType is a match with this Tuple's type.
  /// \tparam   First   The type to test for equivalence with the Tuple.
  /// \tparam   Rest    The potential rest of the types, unused here.
  template <typename First, typename... Rest>
  static constexpr bool isTuple = is_same_v<std::decay_t<First>, SelfType>;

 public:
  /// Alias for the storage type.
  using StorageType = Detail::BasicTuple<Types...>;

  /// Defines the number of elements in the tuple.
  static constexpr size_t elements = StorageType::elements;

  /// Intializes the Tuple with a variadic list of lvalue elements.
  /// \param[in] elements The elements to store in the Tuple.
  VoxxDeviceHost constexpr Tuple(const Types&... elements) noexcept 
  : Storage{elements...} {}

  /// Initializes the Tuple with a variadic list of forwarding reference
  /// elements. This overload is only selcted if the ElementTypes has size one
  /// and the type matches the Tuple, i.e this overload is not selected for
  /// copy and move construction.
  ///  
  /// \param[in] elements     The elements to store in the Tuple.
  /// \tparam    ElementTypes The types of the elements.
  /// \tparam    Enable       Enables this overload if the ElementTypes are not
  ///                         this Tuple's type.
  template <typename... ElementTypes,
            typename    Enable = std::enable_if_t<!isTuple<ElementTypes...>>>
  VoxxDeviceHost constexpr Tuple(ElementTypes&&... elements) noexcept
  : Storage{std::forward<ElementTypes>(elements)...} {}

  /// Copy and move constructs the Tuple. This overload is only selcted if the
  /// TupleType matches this Tuple's type, i.e for copy and move construction.
  ///  
  /// \param[in] other     The other tuple to copy or move.
  /// \tparam    TupleType The type of the other tuple.
  /// \tparam    Enable    Enables this overload if the TupleType is the same as
  ///                      this Tuple's type.
  template <typename TupleType,
            typename Enable = std::enable_if_t<isTuple<TupleType>>>
  VoxxDeviceHost constexpr explicit Tuple(TupleType&& other) noexcept
  : Tuple{std::make_index_sequence<elements>{}, std::forward<TupleType>(other)}
  {}

  /// Returns the underlying storage container, which holds the elements.
  VoxxDeviceHost StorageType& data() noexcept { 
    return Storage; 
  }

  /// Returns a constant reference to the underlying storage container,
  /// which holds the elements.
  VoxxDeviceHost const StorageType& data() const noexcept { 
    return Storage;
  }

  /// Returns the underlying storage container, which holds the elements.
  VoxxDeviceHost volatile StorageType& data() volatile noexcept { 
    return Storage; 
  }

  /// Returns a constant reference to the underlying storage container,
  /// which holds the elements.
  VoxxDeviceHost const volatile StorageType& data() const volatile noexcept { 
    return Storage;
  }

  //==--- Members ----------------------------------------------------------==//
  
  /// Returns the element at position Index.
  /// \tparam   Index   The index of the element to get.
  template <std::size_t Index>
  VoxxDeviceHost constexpr decltype(auto) at() noexcept {
    return Detail::getImpl<Index>(Storage);
  }

  /// Returns the element at position Index.
  /// \tparam   Index   The index of the element to get.
  template <std::size_t Index>
  VoxxDeviceHost constexpr decltype(auto) at() const noexcept {
    return Detail::getImpl<Index>(Storage);
  }

  /// Returns the element at position Index. This overload is selected for a
  /// volatile tuple.
  /// \tparam   Index   The index of the element to get.
  template <std::size_t Index>
  VoxxDeviceHost constexpr decltype(auto) at() volatile noexcept {
    return Detail::getImpl<Index>(Storage);
  }

  /// Returns the element at position Index. This overload is selected for a
  /// volatile tuple.
  /// \tparam   Index   The index of the element to get.
  template <std::size_t Index>
  VoxxDeviceHost constexpr decltype(auto) at() const volatile noexcept {
    return Detail::getImpl<Index>(Storage);
  }

 private:
  StorageType Storage; //!< Storage of the Tuple elements.


  /// This overload of the constructor is called by the copy and move
  /// constructores to get the elements of the \p other Tuple and copy or move
  /// them into this Tuple.
  /// 
  /// \param[in] extractor Used to extract the elements out of \p other.
  /// \param[in] other     The other tuple to copy or move.
  /// \tparam    TupleType The type of the other tuple.
  /// \tparam    Enable    Enables this overload if the TupleType is the same as
  ///                      this Tuple's type.
  template <std::size_t... I, typename TupleType>
  VoxxDeviceHost constexpr explicit
  Tuple(std::index_sequence<I...> extractor, TupleType&& other) noexcept
  : Storage{Detail::getImpl<I>(std::forward<StorageType>(other.data()))...} {}
};

//==--- get ---------------------------------------------------------------==//

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a const lvalue reference.
/// \param[in] tuple The Tuple to get the element from.
/// \tparam    Idx   The index of the element to get from the Tuple.
/// \tparam    Types The types of the Tuple elements.
template <size_t Idx, typename... Types>
VoxxDeviceHost constexpr inline decltype(auto)
get(const Tuple<Types...>& tuple) noexcept {
  return Detail::getImpl<Idx>(tuple.data());
}

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a forwarding reference type.
/// \param[in] tuple The Tuple to get the element from.
/// \tparam    Idx   The index of the element to get from the Tuple.
/// \tparam    Types The types of the Tuple elements.
template <size_t Idx, typename... Types>
VoxxDeviceHost constexpr inline decltype(auto)
get(Tuple<Types...>&& tuple) noexcept {
  return Detail::getImpl<Idx>(std::move(tuple.data()));
}

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a const volatile lvalue reference.
/// \param[in] tuple The Tuple to get the element from.
/// \tparam    Idx   The index of the element to get from the Tuple.
/// \tparam    Types The types of the Tuple elements.
template <size_t Idx, typename... Types>
VoxxDeviceHost constexpr inline decltype(auto)
get(const volatile Tuple<Types...>& tuple) noexcept {
  return Detail::getImpl<Idx>(tuple.data());
}

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a volatile forwarding reference type.
/// \param[in] tuple The Tuple to get the element from.
/// \tparam    Idx   The index of the element to get from the Tuple.
/// \tparam    Types The types of the Tuple elements.
template <size_t Idx, typename... Types>
VoxxDeviceHost constexpr inline decltype(auto)
get(volatile Tuple<Types...>&& tuple) noexcept {
  return Detail::getImpl<Idx>(std::move(tuple.data()));
}

//==--- Meta Functions ------------------------------------------------------==//

/// The tuple_element class get the type of the element at index Idx in a tuple.
/// \tparam   Idx       The index of the element to get type type of.
/// \tparam   TupleType The type of the tuple.
template <std::size_t Idx, typename TupleType>
struct tuple_element {
  /// Returns the type of the element at index Idx.
  using type = decltype(get<Idx>(std::declval<TupleType>()));
};

/// The tuple_element class can provide the type the Nth tuple element.
/// The tuple_element class get the type of the element at index Idx in a tuple.
/// This defines the implementation of the functionaliAty.
/*
template <std::size_t Idx, typename... Types>
struct tuple_element<Idx, Tuple<Types...>> {
 private:
  /// Defines the type of the tuple.
  using TupleType = Tuple<Types...>;
 
 public:
  /// Returns the type of the element at index Idx.
  using type      = decltype(get<Idx>(std::declval<TupleType>()));
};
*/
/// Alias for tuple_element::type.
template <std::size_t Idx, typename TupleType>
using tuple_element_t = typename tuple_element<Idx, TupleType>::type;

/// Defines a struct to concatenate two (maybe more) Tuples types into a single
/// Tuple type. 
///
/// This does not operate on Tuple objects, but rather on Tuple types. This
/// metafuncion is designed to be used to modified the type of the Tuple at 
/// compile time. Essentially this allows a mutable compile-time type 
/// container.
/// 
/// \tparam TupleBase The first tuple to concatenate to.
/// \tparam TupleNext The second Tuple to concatenate to.
template <typename TupleBase, typename TupleNext>
struct TupleCatType;

/// Defines the implementation of the Tuple type concatenation. This simply 
/// creates a new Tuple type using the types of each of the Tuples.
/// \tparam BaseTypes The types of the Tuple being concatenated to.
/// \tparam NewTyeps  The types of the Tuple being concatenated with.
template <typename... BaseTypes, typename... NewTypes>
struct TupleCatType<Tuple<BaseTypes...>, Tuple<NewTypes...>> {
  /// Defines the type of the concatenated Tuple.
  using Type = Tuple<BaseTypes..., NewTypes...>;
};

//==--- Functions ----------------------------------------------------------==//

/// This makes a tuple, and is the interface through which tuples should be
/// created in almost all cases. Example usage is:
///
/// ~~~cpp
/// auto tuple = make_tuple(4, 3.5, "some value");  
/// ~~~
/// 
/// This imlementation decays the types, so it will not create refrence types,
/// i.e:
/// 
/// ~~~cpp
/// int x = 4, y = 5;
/// auto tup = make_tuple(x, y);
/// ~~~
/// 
/// will copy ``x`` and ``y`` and not created a tuple of references to the
/// variables. However, that kind of desired behavious is one of the instances
/// when explicit creation of the tuple is required:
/// 
/// ~~~cpp
/// int x = 0, y = 0.0f;
/// Tuple<int&, float&> tuple = make_tuple(x, y);
/// 
/// // Can modify x and y though tuple:
/// get<0>(tuple) = 4;
/// tuple.at<1>() = 3.5f;
/// 
/// // Can modify tuple values though x and y:
/// x = 0; y = 0.0f;
/// ~~~
/// 
/// \param[in]  values  The values to store in the tuple.
/// \tparam     Types   The types of the values, and which will define the type
///                     of the Tuple.
template <typename... Types>
VoxxDeviceHost constexpr inline decltype(auto)
make_tuple(Types&&... params) noexcept {
  return Tuple<std::decay_t<Types>...>{std::forward<Types>(params)...};
}

} // namespace Voxx
