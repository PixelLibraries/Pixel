//==--- Voxel/Traits/ContainerTraits.hpp ------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  ContainerTraits.hpp
/// \brief This file defines the interface for container traits.
// 
//==------------------------------------------------------------------------==//

#pragma once

namespace Voxx {

/// The ContainerTraits struct defines traits for containers, which can be used
/// to provide generic interfaces for operations on containers of all types.
/// 
/// There are two options for providing traits for a custom container:
/// 
/// 1. Ensure that all the types used by this class are present in the custom
///    container.
///    
/// 2. Specialize this class for the container.
/// 
/// __Note__: Usage of the class __does not__ require the class to be decayed
///           first.
///           
/// ~~~cpp
/// template <typename Container>
/// auto someInterfaceFunction(Container&& container) {
///   if constexpr(ContainerTraits<Container>::isFixedSize) {
///     // Static implementation ...
///   } else {
///     // Dynamic implementation ...
///   }
/// }
/// 
/// \tparam Container The type of the container to get the traits for.
template <typename Container>
struct ContainerTraits {
  //==--- Aliases ----------------------------------------------------------==//
  
  // These are derived from the container:

  /// Define the decayed type of the container.
  using DecayedContainer = std::decay_t<Container>;   
  /// Defines the type of the data elements in the container.
  using DataType         = typename DecayedContainer::DataType;
  /// Defines the size type used by the container.
  using SizeType         = typename DecayedContainer::SizeType;

  // These are derived from the above traits:
  
  /// Defines the decayed type of the data elements. See std::decay docs.
  using DecayedType    = std::decay_t<DataType>;
  /// Defines the value type of the container elements. That is, references, 
  /// const qualifiers, and pointers are removed.
  using ValueType      = std::remove_pointer_t<DecayedType>;
  /// Defines a reference type to the data element, after the data element is
  /// completely decayed (made to be ValueType).
  using Reference      = ValueType&;
  /// Defines a constent reference type to the data element, after the data
  /// element is completely decayed (made to be ValueType).
  using ConstReference = const ValueType&;
  /// Defines a pointer type to the data element, after the data element is
  /// completely decayed (made to be ValueType).
  using Pointer        = ValueType*;
  /// Defines a constent pointer type to the data element, after the data
  /// element is completely decayed (made to be ValueType).
  using ConstPointer   = const ValueType*;

  //==--- Constexpr --------------------------------------------------------==//
  
  /// Returns true if the container has a size which is fixed at compile time.
  static constexpr bool isFixedSize = DecayedContainer::isFixedSize;
  /// Returns the size of the container. This should only be called after a
  /// checking if ``isFixedSize`` is true.
  static constexpr SizeType size() {
    static_assert(DecayedContainer::isFixedSize,
                  "ContainerTraits::size is invalid for dynamic container");

    // Note that we used *elements* to denote the static size so that a
    // size() member can still be provided which can be used on instatnces of
    // the class.
    return DecayedContainer::elements;
  } 
};

} // namespace Voxx