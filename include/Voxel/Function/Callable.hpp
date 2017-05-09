//==--- Voxel/Function/DeviceCallable.hpp ------------------ -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  DeviceCallable.hpp
/// \brief This file defines a wrapper for a generic callable object which can
///        be invoked from device kernels. This provides versions for statically
///        allocating the callable's arguments, as well as dynamically
///        allocating the arguments so that they can be copied back to the host.
//
//==------------------------------------------------------------------------==//

#pragma once

#include "CallableFwd.hpp"
#include "Invoke.hpp"
#include <Voxel/Container/Tuple.hpp>
#include <Voxel/Traits/TypeTraits.hpp>

namespace Voxx     {
namespace Function {

/// Implementation of the static device callable wrapper. This implementation
/// will work on both host and the device.
///  
/// \todo Maybe add copy and move construction/assignment from a dynamic
///       callable.
///       
/// \tparam   CallableT  The type of the callable to wrap.
/// \tparam   Args       The type of the arguments to wrap.
template <typename CallableT, typename... Args>
class Callable {
 public:
  /// Defines the type of the callable.
  using CallableType    = CallableT;
  /// Defines the type of the wrapped arguments.
  using WrappedArgsType = Tuple<Args...>;

  /// The DynamicCallableWrapper is a friend so that we can create dynamic
  /// callable wrappers from this callable wrapper.
  template <typename DynamicCallableT, typename... DynamicArgs>
  friend class DynamicDeviceCallable;

  /// Constructor -- Takes a callable and an arg container and stores them.
  /// We move both here, as we don't want references to functions which may be
  /// out of scope when the callable needs to be invoked (i.e locally defined
  /// functors or lambds).
  /// \param[in]  callable  The callable object to wrap.
  /// \param[in]  args      The arguments for the callable.
  VoxxDeviceHost
  Callable(CallableType&& callable, Args&&... args)
  : WrappedArgs{WrappedArgsType{std::forward<Args>(args)...}},
    CallableObj{std::move(callable)}                         {}

  /// Constructor -- Takes a callable and an arg container and stores them.
  /// We move both here, as we don't want references to functions which may be
  /// out of scope when the callable needs to be invoked (i.e locally defined
  /// functors or lambds).
  /// \param[in]  callable  The callable object to wrap.
  /// \param[in]  args      The arguments for the callable.    
  VoxxDeviceHost
  Callable(CallableType&& callable, WrappedArgsType&& args)
  : WrappedArgs{std::move(args)}, CallableObj(std::move(callable)) {}

  /// Copy constructor -- the callable and arguments are copied.
  /// \param[in]  other   The other wrapper to copy.
  VoxxDeviceHost Callable(const Callable& other)
  : WrappedArgs{other.WrappedArgs}, CallableObj{other.CallableObj} {}

  /// Move constructor -- moves the \p other's callable and arguments.
  /// \param[in]  other   The other wrapper to copy.
  VoxxDeviceHost Callable(Callable&& other) noexcept
  : WrappedArgs{std::move(other.WrappedArgs)},
    CallableObj{std::move(other.CallableObj)} {}

  //==--- Operator overloads -----------------------------------------------==//

  /// Copy assignment -- deleted.
  VoxxDeviceHost Callable& operator=(const Callable&) = delete;

  /// Move assignment -- deleted.
  VoxxDeviceHost Callable& operator=(Callable&&)      = delete;
  
  /// Overload of the call operator to invoke the callable.
  VoxxDeviceHost decltype(auto) operator()() {
    return invoke(std::move(CallableObj), std::move(WrappedArgs));
  }

 private:
  WrappedArgsType WrappedArgs;  //!< The arguments for the callable.
  CallableType    CallableObj;  //!< The callable object.
};

//==--- Functions ----------------------------------------------------------==//

/// This function wraps a callable object in a StaticDeviceCallable if the Args
/// do not contain non const lvalue reference types, or a DynamicDeviceCallable
/// if the args do contain a non const lvalue reference.
/// \param[in]  callable      The callable to create a device version of.
/// \param[in]  args          The arguments for the callable.
/// \tparam     CallableType  The type of the callable.
/// \tparam     Args          The type of the args.
template <typename CallableType, typename... Args>
VoxxDeviceHost constexpr inline decltype(auto)
makeCallable(CallableType&& callable, Args&&... args) {
  using ReturnType = Callable<CallableType, Args...>;
  return ReturnType{std::forward<CallableType>(callable),
                    std::forward<Args>(args)...         };
}

}} // namespace Voxx::Function