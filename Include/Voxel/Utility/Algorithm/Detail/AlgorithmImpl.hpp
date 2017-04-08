//==--- Utility/Algorithm/Detail/AlgorithmImpl.hpp --------- -*- C++ -*- ---==//
//            
//                                Voxel : Utility 
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  AlgorithmImpl.hpp
/// \brief This file provides the implementation of algorithmic functionality.
//
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Utility/Algorithm/AlgorithmFwd.hpp>
#include <tuple>
#include <utility>

namespace Voxx    {
namespace Utility {

//==--- invoke -------------------------------------------------------------==//

template <typename Callable, typename T>
VoxxDeviceHost constexpr inline void 
invoke(Callable&& callable, T&& element) { 
  callable(std::forward<T>(element)); 
}

template <typename Callable, typename T, typename... Ts>
VoxxDeviceHost constexpr inline void 
invoke(Callable&& callable, T&& element, Ts&&... elements) { 
    callable(std::forward<T>(element));
    invoke(std::forward<Callable>(callable), std::forward<Ts>(elements)...);
}

template <typename Callable, typename T>
VoxxDeviceHost constexpr inline void
invoke(Callable&& callable, const T& element) { 
  callable(element); 
}

template <typename Callable, typename T, typename... Ts>
VoxxDeviceHost constexpr inline void 
invoke(Callable&& callable, const T& element, const Ts&... elements) { 
    callable(element);
    invoke(std::forward<Callable>(callable), elements...);
}

namespace Detail {

//==--- forEachImpl std::tuple specializations -----------------------------==//

/// This is a helper implementation for the forEach function to apply a callable
/// object on each element in a tuple. This overload is for a constant reference
/// tuple.
/// \param[in] tuple    A reference to the tuple to invoke the callable on.
/// \param[in] callable The callable object.
/// \tparam    Callable The type of the callable object.
/// \tparam    Ts       The types of the tuple elements.
/// \tparam    I        The indices to get the tuple elements.
template <typename Callable, typename... Ts, size_t... I>
constexpr inline void forEachImpl(const std::tuple<Ts...>&  tuple   ,
                                  Callable&&                callable,
                                  std::index_sequence<I...>         ) {
  invoke(std::forward<Callable>(callable), std::get<I>(tuple)...);
}

/// This is a helper implementation for the forEach function to apply a callable
/// object on each element in a tuple. This overload is for a rvalue-reference
/// tuple.
/// \param[in] tuple    A reference to the tuple to invoke the callable on.
/// \param[in] callable The callable object.
/// \tparam    Callable The type of the callable object.
/// \tparam    Ts       The types of the tuple elements.
/// \tparam    I        The indices to get the tuple elements.
template <typename Callable, typename... Ts, size_t... I>
constexpr inline void forEachImpl(std::tuple<Ts...>&&       tuple   ,
                                  Callable&&                callable,
                                  std::index_sequence<I...>         ) {
  invoke(std::forward<Callable>(callable), std::move(std::get<I>(tuple))...);
}

} // namespace Detail
  
//==--- forEach std::tuple specializations ---------------------------------==//

template <typename Callable, typename... Ts>
constexpr inline void 
forEach(std::tuple<Ts...>&& tuple, Callable callable) {
  Detail::forEachImpl(std::move(tuple)                         ,
                      std::move(callable)                      ,
                      std::make_index_sequence<sizeof...(Ts)>{});
}

template <typename Callable, typename... Ts>
constexpr inline void 
forEach(const std::tuple<Ts...>& tuple, Callable callable) {
  Detail::forEachImpl(tuple                                    ,
                      std::move(callable)                      ,
                      std::make_index_sequence<sizeof...(Ts)>{});
}

}} // namespace Voxx::Utility