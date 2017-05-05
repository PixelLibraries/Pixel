//==--- Voxel/Function/CallableWrapper.hpp ----------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  CallableWrapper.hpp
/// \brief This file defines a wrapper for a callable object which stores the
///        arguments in a tuple, allowing them to be unpacked at the callsite.
//
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Utility/Portability.hpp>

namespace Voxx {

/// This class wraps a callable objects by storing the callable object and the
/// callable's arguments. It overloads the function call operator to allow it to
/// be invoked.
/// 
/// __Note__: This stores the arguments with the types as defined in the
///           callable, i.e if the argument specifies a reference to an object,
///           then a reference type is stored. We leave it up to the
///           implementation which used this class to decay the argument types
///           where appropriate.
///           
/// \todo Test if this will work with pointer to member functions.
/// 
/// \tparam     Callable  The type of the callable to wrap.
/// \tparam     ArgTypes  The type of the arguments to wrap.
template <typename Callable, typename... ArgTypes>
struct CallableWrapper {
  /// Defines the type of the callable.
  using CallableType    = Callable;
  /// Defines the type of the wrapped arguments.
  using WrappedArgsType = Tuple<std::decay_t<ArgTypes>...>;

  /// Constructor -- Takes a callable and an arg container and stores them.
  /// We move both here, as we don't want references to functions which may be
  /// out of scope when the callable needs to be invoked (i.e locally defined
  /// functors or lambds).
  /// \param[in]  callable  The callable object to wrap.
  /// \param[in]  args      The arguments for the callable.    
  DeviceHost CallableWrapper(CallableType&& callable, WrappedArgsType&& args)
  : CallableObj(std::move(callable)), Args(std::move(args)) {}

  /// Copy constructor -- the callable and arguments are copied.
  /// \param[in]  other   The other wrapper to copy.
  DeviceHost CallableWrapper(const CallableWrapper& other)
  : CallableObj(other.CallableObj), WrappedArgsType(other.Args) {}

  /// Move constructor -- moves the \p other's callable and arguments.
  /// \param[in]  other   The other wrapper to copy.
  DeviceHost CallableWrapper(CallableWrapper&& other) noexcept
  : CallableObj(std::move(other.CallableObj)), Args(std::move(other.Args)) {}

  //==--- Operator overloads -----------------------------------------------==//
  
  /// Copy assignment -- deleted.
  DeviceHost CallableWrapper& operator=(const CallableWrapper&) = delete;
  /// Move assignment -- deleted.
  DeviceHost CallableWrapper& operator=(CallableWrapper&&)      = delete;
  
  /// Overload of the call operator to invoke the callable.
  DeviceHost void operator()() {
    Detail::invoke<void>(std::move(CallableObj), std::move(Args));
  }

 private:
  CallableType    CallableObj; //!< The callable object.
  WrappedArgsType Args;        //!< The arguments for the callable.
};

}