//==--- Voxel/Algorithm/AlgorithmFwd.hpp ------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  AlgorithmFwd.hpp
/// \brief This file forward declares the algorithm interface.
//
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Utility/Portability.hpp>
#include <tuple>

namespace Voxx {

/// This function invokes a callable function \p c on the element \p element,
/// and then on the remaining \p elements, when \p element and \p elements are
/// universal reference types.
/// \param[in]  callable  The callable object to invoke.
/// \param[in]  element   The element to invoke the callable on.
/// \param[in]  elements  Additional elements to invoke the callable on.
/// \tparam     Callable  The type of the callable object.
/// \tparam     T         The type of the object to pass to the callable.
/// \tparam     Ts        The types of the additional elements to invoke on.
template <typename Callable, typename T, typename... Ts>
VoxxDeviceHost constexpr inline void 
invoke(Callable&& callable, T&& element , Ts&&... elements);

/// This function invokes a callable function \p c on the element \p element,
/// and then on the remaining \p elements, when \p element and \p elements are
/// constant reference types.
/// \param[in]  callable  The callable object to invoke.
/// \param[in]  element   The element to invoke the callable on.
/// \param[in]  elements  Additional elements to invoke the callable on.
/// \tparam     Callable  The type of the callable object.
/// \tparam     T         The type of the object to pass to the callable.
/// \tparam     Ts        The types of the additional elements to invoke on.
template <typename Callable, typename T, typename... Ts>
VoxxDeviceHost constexpr inline void 
invoke(Callable&& callable, const T& element , const Ts&... elements);

/*
/// This function applies the callable \p callable to each element of the \p
/// container, when the \p container is an universal reference type.
/// \param[in]  container The container to apply the callable to.
/// \paran[in]  callable  The callable object to invoke on the elements.
/// \tparam     Callable  The type of the callable object.
/// \tparam     Container The type of the container.
template <typename Callable, typename Container>
VoxxDeviceHost constexpr inline void
forEach(Container&& container, Callable callable);

/// This function applies the callable \p callable to each element of the \p
/// container, when the \p container is a constant reference type.
/// \param[in]  container The container to apply the callable to.
/// \paran[in]  callable  The callable object to invoke on the elements.
/// \tparam     Callable  The type of the callable object.
/// \tparam     Container The type of the container.
template <typename Callable, typename Container>
VoxxDeviceHost constexpr inline void
forEach(const Container& container, Callable callable);
*/

/// This function applies the callable \p callable to each element of the \p
/// tuple, when the \p tuple is an universal reference type.
/// \param[in]  tuple     The tuple to apply the callable to.
/// \paran[in]  callable  The callable object to invoke on the tuple elements.
/// \tparam     Callable  The type of the callable object.
/// \tparam     Ts        The types of the tuple elements.
template <typename Callable, typename... Ts>
constexpr inline void 
forEach(std::tuple<Ts...>&& tuple, Callable callable);

/// This function applies the callable \p callable to each element of the \p
/// tuple, when the \p tuple is an constant reference type.
/// \param[in]  tuple     The tuple to apply the callable to.
/// \paran[in]  callable  The callable object to invoke on the tuple elements.
/// \tparam     Callable  The type of the callable object.
/// \tparam     Ts        The types of the tuple elements.
template <typename Callable, typename... Ts>
constexpr inline void 
forEach(const std::tuple<Ts...>& tuple, Callable callable);

} // namespace Voxx