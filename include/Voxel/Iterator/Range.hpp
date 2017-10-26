//==--- Voxel/Iterator/Range.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  Range.hpp
/// \brief This file defines a range which can be used to create a python like
///        range to iterator over.
// 
//==------------------------------------------------------------------------==//

#ifndef VOXX_ITERATOR_RANGE_HPP
#define VOXX_ITERATOR_RANGE_HPP

#include <Voxel/Utility/Portability.hpp>
#include <cmath>
#include <type_traits>

namespace Voxx {

/// The Range class defines a utility class which allows a simpler syntax
/// for looping. It allows a python like range based for-loop. It is wrapped
/// by the ``range()`` function, which is the interface which should be used
/// to create a range in most cases. Example usage is the following:
/// 
/// ~~~cpp
/// // i = [0 ... 100)
/// for (auto i : range(100)) {
///   // Use i
/// }
/// 
/// // i = [34 .. 70)
/// for (auto i : range(34, 70)) {
///   // Use i
/// }
/// 
/// // i = [23 .. 69) step = 3
/// for (auto i : range(23, 69, 3)) {
///   // Use i 
/// }
/// 
/// The type of the range elements are defined by the type passed to the range
/// creation function (i.e pass a float to get a range of floats):
/// 
/// ~~~cpp
/// for (auto i : range(0.1f, 0.6f, 0.1f)) {
///   // Use i
/// }
/// 
/// \tparam T The type of the range data.
template <typename T>
class Range {
 private:
  //==--- Contants ---------------------------------------------------------==//
  
  /// Defines if the type of the Range is a floating point type.
  static constexpr auto isFloatingPoint = std::is_floating_point_v<T>;

  //==--- Classes ----------------------------------------------------------==//
  
  /// The Iterator class defines an iterator for iterating over a range.
  /// \tparam IsConst   If the iterator is constant.
  template <bool IsConst>
  class Iterator {
   public:
    //==--- Aliases --------------------------------------------------------==//
    
    /// Defines the type of the iterator.
    using SelfType  = Iterator;
    /// Defines the type of the iterator data.
    using ValueType = T;
    /// Defines the type of a reference.
    using Reference = std::conditional_t<IsConst, const T&, T&>;
    /// Defines the type of a pointer.
    using Pointer   = std::conditional_t<IsConst, const T*, T*>;
    /// Defines the category of the iterator.
    //using iterator_category = std::forward_iterator_tag;

    /// Constructor -- sets the value of the iterator and the step size.
    /// \param[in]  value   The value for the iterator.
    /// \param[in]  step    The size of the steps for the iterator.
    VoxxDeviceHost Iterator(ValueType value, ValueType step)
    : Value(value), Step(step) {}

    //==--- Operator overloads ---------------------------------------------==//
    
    /// Overload of increment operator to move the iterator forward by the step
    /// size. This overload is for the postfix operator and returns the old
    /// value of the iterator.
    VoxxDeviceHost constexpr SelfType operator++() { 
      SelfType i = *this; Value += Step; return i;
    }

    /// Overload of increment operator to move the iterator forward by the step
    /// size. This overload is for the prefix operator and returns the updated
    /// iterator.
    VoxxDeviceHost constexpr SelfType operator++(int unused) {
      Value += Step; return *this;
    }

    /// Returns a reference to the value of the iterator.
    VoxxDeviceHost constexpr Reference operator*()  { return Value;  }
    /// Returns a pointer to the value of the iterator.
    VoxxDeviceHost constexpr Pointer   operator->() { return &Value; }

    /// Overload of the equality operator to check if two iterators are
    /// equivalent. This returns true if the value of this iterator is greater
    /// than the or equal to the value of \p other.
    /// \param[in]  other   The other iterator to compare with.
    VoxxDeviceHost constexpr bool operator==(const SelfType& other) {
      return Value >= other.Value;
    }

    /// Overload of the inequality operator to check if two iterators are
    /// not equivalent. This returns true if the value of this iterator is less
    /// than the value of \p other.
    /// \param[in]  other   The other iterator to compare with.
    VoxxDeviceHost constexpr bool operator!=(const SelfType& other) {
      return Value < other.Value;
    }

   private:
    ValueType Value; //!< The current value of the range iterator.
    ValueType Step;  //!< The step size of the range iterator.
  };

 public:
  //==--- Aliases ----------------------------------------------------------==//
  
  /// Defines the type of a constant iterator.
  using ConstIterator    = Iterator<true>;
  /// Defines the type of a non-constant iterator.
  using NonConstIterator = Iterator<false>;

  //==--- Con/destruction --------------------------------------------------==//
  
  /// Constructor -- creates the range.
  /// \param[in]  min   The minimum (start) value for the range.
  /// \param[in]  max   The maximum (end) value for the range.
  /// \param[in]  step  The step size for the range.
  VoxxDeviceHost Range(T min, T max, T step)
  : Min(min), Max(max), Step(step) {}

  //==--- Methods ----------------------------------------------------------==//

  /// Gets a non constant iterator to the beginning of the range.
  VoxxDeviceHost decltype(auto) begin() {
    return NonConstIterator(Min, Step);
  }
  
  /// Gets a non constant iterator to the end of the range.
  VoxxDeviceHost decltype(auto) end() {
   return NonConstIterator(Max, Step);   
  }

  /// Gets a onstant iterator to the beginning of the range.
  VoxxDeviceHost decltype(auto) begin() const {
    return ConstIterator(Min, Step);
  }
  
  /// Gets a non constant iterator to the end of the range.
  VoxxDeviceHost decltype(auto) end() const {
   return ConstIterator(Max, Step);   
  }

  /// Returns the size (number of elements) in the range. This method is not
  /// high performance for non integral ranges since ensuring correctness is
  /// difficult due to precision errors. If an incorrect case is found, please
  /// add a test case to the test suite.
  /// 
  /// For integral types the performance is much better.
  VoxxDeviceHost std::size_t size() const noexcept {
    const auto result = (Max - Min) / Step;
    if constexpr (isFloatingPoint) {
      return std::fmod(Max - Min, Step) > Step * 0.5f
        ? std::floor(result) : std::nearbyint(result);
    }
    return result;
  }

  /// Returns true if the range is divisible into \p parts parts.
  VoxxDeviceHost bool isDivisible(std::size_t parts = 2) const {
    return (size() % parts) == 0;
  }

  T Min;  //!< The minimum value in the range.
  T Max;  //!< The maximum value in the range.
  T Step; //!< The step size for iterating over the range.
};

//==--- Interface ----------------------------------------------------------==//

/// Creates a range from 0 to \p end, using a step size of 1.
/// \param[in]  end   The end value for the range
/// \tparam     T     The type of the range data.
template <typename T>
VoxxDeviceHost constexpr inline decltype(auto) range(T end) {
  return Range<T>(T(0), end, T(1));
}

/// Creates a range from \p start to \p end, using a step size of \p step.
/// \param[in]  start The staring value of the range.
/// \param[in]  end   The end value for the range
/// \param[in]  step  The step size of the range.
/// \tparam     T     The type of the range data.
template <typename T> 
VoxxDeviceHost constexpr inline decltype(auto)
range(T start, T end, T step = T(1)) {
  return Range<T>(start, end, step);
}

} // namespace Voxx

#endif // VOXX_ITERATOR_ITERATORS_HPP