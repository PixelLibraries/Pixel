//==--- Voxel/Heterogeneous/Detail/LaunchImpl.hpp ---------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  LaunchImpl.hpp
/// \brief This file implements functionality to launch a callable object from
///        the host, on the device.
// 
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Function/Callable.hpp>
#include <Voxel/Function/DynamicDeviceCallable.hpp>
#include <Voxel/Utility/Portability.hpp>
#include <Voxel/Utility/Debug.hpp>
#include <type_traits>

namespace Voxx   {
namespace Hetero {
namespace Detail {

//==--- Traits -------------------------------------------------------------==//

/// Returns true of the parameter pack containes a non const lvalue reference,
/// and hence may vary.
/// \tparam Args  The args to check the variability of.
template <typename... Args>
static constexpr bool mayVary =
  Traits::has_non_const_lvalue_reference_v<Args...>;

//==--- Device related -----------------------------------------------------==//

/// This function invokes the \p callable on the device.
/// \param[in]  callable        The wrapped callable object to invoke.
/// \tparam     CallableWrapper The type of the wrapped callable object.
template <typename CallableWrapper>
__global__ void deviceLaunch(CallableWrapper callable) {
  callable();
}

//==--- Launch Definitions -------------------------------------------------==//

/// Defines if the device much launch with arguments which may be modified on
/// the device (i.e references), or ones which will not be.
enum LaunchType : uint8_t {
  Static  = 0x00,   //!< Launch with statically allocated arguments.
  Dynamic = 0x01    //!< Launch with dynamically allocated arguments.
};

/// This is a helper struct which is used to correctly invoked a wrapped
/// callable type on the device. It is specialized for the cases that the
/// arguments to the callable contain non const lvalue references (and hence
/// may change), and that the arguments dont't change.
/// 
/// \tparam   WrappedCallable   The type of the wrapped callable object.
/// \tparam   LaunchImpl        The implementation to launch: whether the
///                             callable's args can be modified or not.
template <typename WrappedCallable, uint8_t LaunchImpl>
struct LaunchHelper;

/// Alias for a launch helper which does not allocate and modify arguments on
/// the device.
/// \tparam CallableType  The type of the callable.
/// \tparam Args      The types of the args.
template <typename CallableType, typename... Args>
using StaticLauncher = 
  LaunchHelper<Function::Callable<CallableType, Args...>, LaunchType::Static>;

/// Alias for a launch helper which does allocate and may modify arguments on
/// the device.
/// \tparam Callable  The type of the callable.
/// \tparam Args      The types of the args.
template <typename CallableType, typename... Args>
using DynamicLauncher = LaunchHelper<
  Function::DynamicDeviceCallable<CallableType, Args...>, LaunchType::Dynamic>;

/// Alias for the type of device callable closure to use. If the arguments
/// container a non const lvalue reference then they may vary (i.e invoking the
/// callable may set the reference) and hence we need to allocate them on the
/// device and copy them back into the original args after the call. This alias
/// defines the type of device closure depending on the above.
/// 
/// \tparam Callable  The type of the callable.
/// \tparam Args      The types of the args.
template <typename Callable, typename... Args>
using DeviceLauncher = std::conditional_t<mayVary<Args...>,
  DynamicLauncher<Callable, Args...>, StaticLauncher<Callable, Args...>>;

//==--- Launch implementations ---------------------------------------------==//

/// This is a helper class for launching a callable object on the GPU, where the
/// arguments to the callable object are __not__ dynamically allocated on the
/// device. This allows them to to copied onto the device more efficiently,
/// but they cannot be retrieved once they are on the device.
/// \tparam WrappedCallable The type of the wrapped callable object.
template <typename WrappedCallable>
struct LaunchHelper<WrappedCallable, LaunchType::Static> {
  /// Defines the type of the callable.
  using CallableType    = typename WrappedCallable::CallableType;
  /// Defines the type of the wrapped arguments.
  using WrappedArgsType = typename WrappedCallable::WrappedArgsType;

  /// Launches the \p callable object forwarding its arguments.
  /// \tparam DimType   The type of the dim object to specify the sizes of the
  ///                   computational domain.
  /// \tparam Args      The types of the arguments.
  template <typename DimType, typename... Args>
  static void launch(DimType&&      gridDim,
                     DimType&&      blockDim,
                     CallableType&& callable,
                     Args&&...      args    ) {
//    WrappedCallable wrappedCallable(std::move(callable)        ,
//                                    std::forward<Args>(args)...);
    deviceLaunch<<<gridDim, blockDim>>>(
      WrappedCallable{std::move(callable), std::forward<Args>(args)...});
    VoxxCudaErrorCheck(cudaDeviceSynchronize());
  }
};

/// This is a helper class to launch a callable object on the GPU where __any__
/// of the arguments for the lambda are non-const lvalue references. If that is
/// the case, we have to assume that the lambda __may__ modify the non-const 
/// lvalue references, so we need to synamically allocate them on the device,
/// allow the kernel call to copy the pointer to the arguments to the device,
/// and then copy the modified arguments back into any of the non const lvalue
/// reference arguments.
/// \tparam WrappedCallable The type of the wrapped callable object.
template <typename WrappedCallable>
struct LaunchHelper<WrappedCallable, LaunchType::Dynamic> {
  /// Defines the type of the callable.
  using CallableType    = typename WrappedCallable::CallableType;
  /// Defines the type of the wrapped arguments.
  using WrappedArgsType = typename WrappedCallable::WrappedArgsType;

  /// Launches the \p callable object on the device.
  /// \tparam DimType   The type of the dim object to specify the sizes of the
  ///                   computational domain.
  /// \tparam Args      The types of the arguments.
  template <typename DimType, typename... Args>
  static void launch(DimType&&      gridDim,
                     DimType&&      blockDim,
                     CallableType&& callable,
                     Args&&...      args    ) {
    /// Create a version which we can copy the modified args from.
    WrappedCallable wrappedCallable(std::move(callable), 
                                    std::forward<Args>(args)...);

    deviceLaunch<<<gridDim, blockDim>>>(wrappedCallable);
    VoxxCudaErrorCheck(cudaDeviceSynchronize());

    wrappedCallable.setOtherArgs(std::forward<Args>(args)...);
  }
};

}}} // namespace Voxx::Hetero;:Detail