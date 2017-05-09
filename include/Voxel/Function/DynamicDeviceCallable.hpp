//==--- Voxel/Function/DynamicDeviceCallable.hpp ----------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  DynamicDeviceCallable.hpp
/// \brief This file defines wrapper classes for callable objects which allocate
///        memory for the arguments of the callable. This behaviour is desired
///        if the arguments are references and the callable will be executed on
///        the device, or potentially if the callable wrapper should be passed
///        between threads.
//
//==------------------------------------------------------------------------==//

#pragma once

#include "CallableFwd.hpp"
#include "Invoke.hpp"
#include <Voxel/Container/Tuple.hpp>
#include <Voxel/Heterogeneous/Utils.hpp>
#include <Voxel/Traits/TypeTraits.hpp>

namespace Voxx     {
namespace Function {

/// Implementation of a dynamic device callable wrapper which allocates its
/// arguments on the device.
/// \tparam   CallableT  The type of the callable to wrap.
/// \tparam   Args       The type of the arguments to wrap.
template <typename CallableT, typename... Args>
class DynamicDeviceCallable {
 public:
  /// Defines the type of the callable.
  using CallableType    = CallableT;
  /// Defines the type of the wrapped arguments. References are removed because
  /// the behaviour is mimiced by copying to and from the device.
  using WrappedArgsType = Tuple<std::remove_reference_t<Args>...>;

  /// The StaticDeviceCallable is a friend class so we can construct and copy
  /// from it.
  template <typename DefaultCallable, typename... ArgTypes>
  friend class Callable;

  /// Constructor -- Takes a callable and and arguments for the callable, and
  /// moves the callable into the container and stores copies of the arguments
  /// on the device.
  /// \param[in]  callable  The callable object to wrap.
  /// \param[in]  args      The arguments for the callable.
  /// \tparam     ArgTypes  The types of the arguments.
  template <typename... ArgTypes>
  DynamicDeviceCallable(CallableType&& callable, ArgTypes&&... args)
  :   WrappedArgs{nullptr}, CallableObj{std::move(callable)}, MustFree{true} {
    initializeOnDevice(
      WrappedArgsType{std::remove_reference_t<ArgTypes>(args)...});
  }

  /// Constructor -- Creates a wrapped callable from a StaticDeviceCallable by
  /// copying the \p other's callable and arguments onto the device.
  /// \param[in]  other     The other callable wrapper to construct from.
  /// \tparam     ArgTypes  The type of the other wrapper's args.
  template <typename... ArgTypes>
  DynamicDeviceCallable(const Callable<CallableType, ArgTypes...>& other)
  :   WrappedArgs{nullptr}, CallableObj{other.CallableObj}, MustFree{true} {
    intiializeOnDevice(other.WrappedArgs);
  }

  /// Copy constructor -- the callable and arguments are copied. This assumes
  /// that \p other will go out of scope __after__ this newly created one, and
  /// hence that we (and __not it__) will free the device allocated arguments.
  /// \param[in]  other   The other wrapper to copy.
  VoxxDeviceHost DynamicDeviceCallable(const DynamicDeviceCallable& other)
  : WrappedArgs{other.WrappedArgs},
    CallableObj{other.CallableObj},
    MustFree{false}               {}

  /// Move constructor -- moves the \p other's callable and arguments. This sets
  /// that this wrapper will free the device allocated args and not \p other.
  /// \param[in]  other   The other wrapper to copy.
  VoxxDeviceHost DynamicDeviceCallable(DynamicDeviceCallable&& other) noexcept
  :   WrappedArgs{std::move(other.WrappedArgs)},
      CallableObj{std::move(other.CallableObj)},
      MustFree{true}                           {
    other.MustFree = false;
  }

  /// Destructor -- frees the device allocated arguments, if we are on the host,
  /// and if were are still responsible for freeing the device allocated
  /// arguments, otherwise we do nothing.
  VoxxDeviceHost ~DynamicDeviceCallable() noexcept {
#if !defined(VoxxCudaDeviceCompile)
    if (MustFree && WrappedArgs != nullptr)
      Hetero::Device::free(&WrappedArgs); 
#endif
  } 

  //==--- Operator overloads -----------------------------------------------==//
  
  /// Copy assignment -- deleted.
  VoxxDeviceHost
  DynamicDeviceCallable& operator=(const DynamicDeviceCallable&) = delete;

  /// Move assignment -- deleted.
  VoxxDeviceHost
  DynamicDeviceCallable& operator=(DynamicDeviceCallable&&)      = delete;

  /// Assignment from a StaticDeviceCallable. We allow this by copying the
  /// \p other's callable and arguments.
  /// \param[in]  other     The other callable wrapper to construct from.
  /// \tparam     ArgTypes  The type of the other wrapper's args.
  template <typename... ArgTypes>
  DynamicDeviceCallable&
  operator=(const Callable<CallableType, ArgTypes...>& other) {
    CallableObj = other.CallableObj;
    initializeOnDevice(other.WrappedArgs);
  }

  //==--- Methods ----------------------------------------------------------==//
  
  /// Overload of the call operator to invoke the callable, passing a reference
  /// to the device allocated arguments.
  /// This returns whatever the callable returns, but really we should make it
  /// return void.
  VoxxDeviceHost decltype(auto) operator()() {
    return invoke(std::move(CallableObj), *WrappedArgs);
  }

  /// Sets the \p args, where for each of the \p args, if the arg is a
  /// non-const lvalue reference then it is set to the value from WrappedArgs,
  /// otherwise it's current value is not changed.
  /// \param[in]  args        The args to update.
  /// \tparam     UpdateArgs  The type of the args to update.
  template <typename... ArgTypes>
  void setOtherArgs(ArgTypes&&... args) {
    using Indexer = std::make_index_sequence<sizeof...(ArgTypes)>;
    setOtherArgsImpl(copyArgsFromDevice()                               ,
                     Tuple<ArgTypes...>(std::forward<ArgTypes>(args)...),
                     Indexer{}                                          );
  }

 private:
  WrappedArgsType* WrappedArgs; //!< A pointer to device allocated arguments.
  CallableType     CallableObj; //!< The callable object.
  bool             MustFree;    //!< If the allocated arguments must be freed.

  /// Initializes the Args member on the device, given a set of wrapped
  /// arguments.
  /// \param[in]  wrappedArgs   The wrapped arguments to use to initialize the
  ///                           device side arguments.
  /// \tparam     ArgWrapper    The type of the wrapped arguments.
  template <typename ArgWrapper>
  void initializeOnDevice(ArgWrapper&& wrappedArgs) noexcept {
    Hetero::Device::allocate(&WrappedArgs, sizeof(WrappedArgsType));
    Hetero::copyToDevice(&wrappedArgs, WrappedArgs, sizeof(WrappedArgsType));
  }

  /// Copies the device allocated args back to the host.
  /// This returns a WrappedArgsType container with the values filled by the
  /// updated values of the arguments on the device.
  WrappedArgsType copyArgsFromDevice() const noexcept {
    WrappedArgsType hostArgs;
    Hetero::copyToHost(WrappedArgs, &hostArgs, sizeof(WrappedArgsType));
    return hostArgs;
  }

  /// Updates the set of \p mutableValues using the \p constValues, if the type
  /// of a specific \p mutableValues value is a non-const lvalue reference,
  /// otherwise the specific value is left unchanged.
  /// \param[in]  constValues     The values to use to update with.
  /// \param[in]  mutableValues   The values to potentially modify. 
  /// \param[in]  extractor       Used extract the elements.
  /// \tparam     ConstTuple      A tuple containing the constant value types.
  /// \tparam     MutableTuple    A tuple containing the mutable value types.
  /// \tparam     I               The values of the indices.
  template <typename ConstTuple, typename MutableTuple, std::size_t... I>
  static constexpr void
  setOtherArgsImpl(ConstTuple&&              constValues  ,
                   MutableTuple&&            mutableValues,
                   std::index_sequence<I...> extractor    ) noexcept {
    ConstTuple wrapper{
      maybeModify(get<I>(std::forward<ConstTuple>(constValues)),
                  get<I>(mutableValues)                       )...};
  }

  /// This takes two values, and sets \p modifyable to \p modifier if the type
  /// of \p modifyable is a non-const lvalue reference.
  /// \param[in]  modifier    The type to use to modify with.
  /// \param[in]  modifyable  The type to potentially modify.
  /// \tparam     Modifier    Type type of the modifier variable.
  /// \tparam     Modifyable  The type of the modifyable variable.
  /// This returns the \p modifier variable, which the compiler should detect
  /// is unused where returned and remove it.
  template <typename Modifier, typename Modifyable>
  static constexpr decltype(auto)
  maybeModify(Modifier modifier, Modifyable&& modifyable) noexcept {
    if constexpr (Traits::is_non_const_lvalue_reference_v<Modifyable>)
      modifyable = modifier;
    return modifier;
  }
}; 

//==--- Functions ----------------------------------------------------------==//

/// This function wraps a callable and its args into a dynamic device callable.
/// \param[in]  callable      The callable to create a device version of.
/// \param[in]  args          The arguments for the callable.
/// \tparam     CallableType  The type of the callable.
/// \tparam     Args          The type of the args.
template <typename CallableType, typename... Args>
VoxxDeviceHost constexpr inline decltype(auto)
makeDynamicDeviceCallable(CallableType&& callable, Args&&... args) {
  using ReturnType = DynamicDeviceCallable<CallableType, Args...>;
  return ReturnType{std::forward<CallableType>(callable),
                    std::forward<Args>(args)...         };
}

}} // namespace Voxx::Function