//==--- Voxel/Function/Invoke.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  Invoke.hpp
/// \brief This file defines functionality to invoke a callable object. 
//
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Container/Tuple.hpp>
#include <Voxel/Traits/TypeTraits.hpp>
#include <Voxel/Utility/Portability.hpp>
#include <utility>

namespace Voxx   {
namespace Detail {

/// This function invokes the \p callable, unwrapping the wrapped \p wrappedArgs
/// into the funtion call.
/// \param[in] callable     The callable to invoke.
/// \param[in] wrappedArgs  The wrapped up arguments for the function.
/// \param[in] indexer      Used to facilitate the expansion of the args.
/// \tparam    ReturnType   The return type of the functor.
/// \tparam    Callable     The type of the \p callable to execute.
/// \tparam    WrappedArgs  The type of the wrapped args container.
/// \tparam    I            The indices to extract the args.
template <typename CallableType, typename WrappedArgs, std::size_t... I>
VoxxDeviceHost decltype(auto)
invokeImpl(CallableType&&            callable   ,
           WrappedArgs&&             wrappedArgs, 
           std::index_sequence<I...> indexer    ) noexcept {
  return callable(get<I>(std::forward<WrappedArgs>(wrappedArgs))...); 
}

} // namespace Detail

/// This function is the interface for invoking a wrapped callable object on
/// the device. It forwards to the implementation which extracts the wrapped
/// arguments and expands them into the function call.
/// \param[in] callable     The callable object to invoke.
/// \param[in] args         The wrapped up arguments for the function.
/// \tparam    CallableType The type of the \p callable to invoke.
/// \tparam    WrappedArgs  The type of the wrapped args container.
template <typename CallableType, typename WrappedArgs>
VoxxDeviceHost constexpr inline decltype(auto)
invoke(CallableType&& callable, WrappedArgs&& args) noexcept {
  constexpr auto numArgs = std::decay_t<WrappedArgs>::elements;
  return Detail::invokeImpl(std::forward<CallableType>(callable),
                            std::forward<WrappedArgs>(args)     ,
                            std::make_index_sequence<numArgs>() );
}

} // namespace Voxx
