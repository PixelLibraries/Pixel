//==--- Voxel/Algorithm/Detail/AlgorithmImpl.hpp ----------- -*- C++ -*- ---==//
//            
//                                    Voxel
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

#include <Voxel/Container/Tuple.hpp>
#include <utility>

namespace Voxx {

/// This namespace implements algorithmic functionality.
namespace Detail {

//==--- forEachImpl std::tuple specializations -----------------------------==//


/// The Apply struct is just a wrapper struct which invokes the functor on the
/// element with the given arguments.
struct Apply {
  /// This applies the functor to the \p element and \p args. It's just a wrapper
  /// which allows the functor to be applied to all tuple elements.
  /// \param[in]  functor   The functor to apply.
  /// \param[in]  element   The element to apply the functor to.
  /// \param[in]  args      Additionaly arguments for the functor.
  /// \tparam     Functor   The type of the functor.
  /// \tparam     Element   The type of the element.
  /// \tparam     Args      The type of the arguments.
  template <typename Functor, typename Element, typename... Args>
  VoxxDeviceHost constexpr Apply(Functor&& functor,
                                 Element&& element,
                                 Args&&... args   ) {
    functor(std::forward<Element>(element), std::forward<Args>(args)...);
  }
};

/// This is a dummy function which allows a function to be expanded out.
/// \tparam  Ts   The types to expand out.
template <typename... Ts>
VoxxDeviceHost constexpr inline void dummy(Ts&&...) {}

/// Implementation of forEach for the Tuple type, when the tuple type is an
/// rvalue reference.
/// \param[in]  tuple   The tuple to apply forEach to.
/// \param[in]  functor The functor to apply to each tuple element.
/// \param[in]  unpack  Used to unpack the tuple.
/// \param[in]  args    Additional arguments for the functor.
/// \tparam     Functor The type of the functor.
/// \tparam     Ts      The types of the tuple elements.
/// \tparam     I       Accessor for the tuple indices.
/// \tparam     Args    The type of the additional functor args.
template <typename Functor, typename... Ts, size_t... I, typename... Args>
VoxxDeviceHost constexpr inline void
forEachImpl(Tuple<Ts...>&&            tuple  ,
            Functor&&                 functor,
            std::index_sequence<I...> unpack ,
            Args&&...                 args   ) {
  dummy(Apply(std::forward<Functor>(functor), 
              get<I>(std::move(tuple)), args...)...);
}

/// Implementation of forEach for the Tuple type, when the tuple type is a
/// non-constant lvalue reference.
/// \param[in]  tuple   The tuple to apply forEach to.
/// \param[in]  functor The functor to apply to each tuple element.
/// \param[in]  unpack  Used to unpack the tuple.
/// \param[in]  args    Additional arguments for the functor.
/// \tparam     Functor The type of the functor.
/// \tparam     Ts      The types of the tuple elements.
/// \tparam     I       Accessor for the tuple indices.
/// \tparam     Args    The type of the additional functor args.
template <typename Functor, typename... Ts, size_t... I, typename... Args>
VoxxDeviceHost constexpr inline void
forEachImpl(Tuple<Ts...>&             tuple  ,
            Functor&&                 functor,
            std::index_sequence<I...>        ,
            Args&&...                 args   ) {
  dummy(Apply(std::forward<Functor>(functor), get<I>(tuple), args...)...);
}

/// Implementation of forEach for the Tuple type, when the tuple type is a
/// constant lvalue reference.
/// \param[in]  tuple   The tuple to apply forEach to.
/// \param[in]  functor The functor to apply to each tuple element.
/// \param[in]  unpack  Used to unpack the tuple.
/// \param[in]  args    Additional arguments for the functor.
/// \tparam     Functor The type of the functor.
/// \tparam     Ts      The types of the tuple elements.
/// \tparam     I       Accessor for the tuple indices.
/// \tparam     Args    The type of the additional functor args.
template <typename Functor, typename... Ts, size_t... I, typename... Args>
VoxxDeviceHost constexpr inline void
forEachImpl(const Tuple<Ts...>&      tuple  ,
            Functor&&                functor,
            std::index_sequence<I...>       ,
            Args&&...                args   ) {
  dummy(Apply(std::forward<Functor>(functor), get<I>(tuple), args...)...);
}

} // namespace Detail

/// The Unroll struct invokes a callable abject Amount times in an unrolled
/// manner.
/// \tparam Amount  The amount of unrolling to do.
template <std::size_t Amount>
struct Unroll : Unroll<Amount - 1> {
  /// Defines the type of the case class which invokes at the previous level.
  using PreviousLevel = Unroll<Amount - 1>;

  /// Constructor -- passes the \p functor and \p args to the previous level
  /// to invoke, and then invokes at this level.
  /// \param[in]  functor   The functor to invoke.
  /// \param[in]  args      The arguments to pass to the functor.
  /// \tparam     Functor   The type of the functor to invoke.
  /// \tparam     Args      The type of the arguments to invoke with.
  template <typename Functor, typename... Args>
  VoxxDeviceHost Unroll(Functor&& functor, Args&&... args)
  : PreviousLevel(std::forward<Functor>(functor),
                  std::forward<Args>(args)...   ) {
    functor(Amount - 1, std::forward<Args>(args)...);
  }
};

/// Specialization of the unrolling class to terminate the urolling at the
/// lowest level.
template <>
struct Unroll<1> {
  /// Invokes the functor with the given args.
  /// \param[in]  functor   The functor to invoke.
  /// \param[in]  args      The arguments to pass to the functor.
  /// \tparam     Functor   The type of the functor to invoke.
  /// \tparam     Args      The type of the arguments to invoke with.
  template <typename Functor, typename... Args>
  VoxxDeviceHost Unroll(Functor&& functor, Args&&... args) {
    functor(0, std::forward<Args>(args)...);
  }
};

//==--- forEach Tuple specializations --------------------------------------==//

/// Overload of forEach for a Tuple type, when the tuple is a non-const lvalue
/// reference.
/// \param[in]  tuple   The tuple to apply forEach to.
/// \param[in]  functor The functor to apply to each tuple element.
/// \param[in]  args    Additional arguments for the functor.
/// \tparam     Functor The type of the functor.
/// \tparam     Ts      The types of the tuple elements.
/// \tparam     Args    The type of the additional functor args.
template <typename Functor, typename... Ts, typename... Args>
VoxxDeviceHost constexpr inline void 
forEach(Tuple<Ts...>& tuple, Functor&& functor, Args&&... args) {
  Detail::forEachImpl(tuple                                    ,
                      std::forward<Functor>(functor)           ,
                      std::make_index_sequence<sizeof...(Ts)>{},
                      std::forward<Args>(args)...              );
}

/// Overload of forEach for a Tuple type, when the tuple is an rvalue refernece.
/// \param[in]  tuple   The tuple to apply forEach to.
/// \param[in]  functor The functor to apply to each tuple element.
/// \param[in]  args    Additional arguments for the functor.
/// \tparam     Functor The type of the functor.
/// \tparam     Ts      The types of the tuple elements.
/// \tparam     Args    The type of the additional functor args.
template <typename Functor, typename... Ts, typename... Args>
VoxxDeviceHost constexpr inline void 
forEach(Tuple<Ts...>&& tuple, Functor&& functor, Args&&... args) {
  Detail::forEachImpl(std::move(tuple)                         ,
                      std::forward(functor)                    ,
                      std::make_index_sequence<sizeof...(Ts)>{},
                      std::forward<Args>(args)...              );
}

/// Overload of forEach for a Tuple type, when the tuple is a const lvalue
/// reference.
/// \param[in]  tuple   The tuple to apply forEach to.
/// \param[in]  functor The functor to apply to each tuple element.
/// \param[in]  args    Additional arguments for the functor.
/// \tparam     Functor The type of the functor.
/// \tparam     Ts      The types of the tuple elements.
/// \tparam     Args    The type of the additional functor args.
template <typename Functor, typename... Ts, typename... Args>
VoxxDeviceHost constexpr inline void 
forEach(const Tuple<Ts...>&& tuple, Functor&& functor, Args&&... args) {
  Detail::forEachImpl(tuple                                    ,
                      std::forward(functor)                    ,
                      std::make_index_sequence<sizeof...(Ts)>{},
                      std::forward<Args>(args)...              );
}

//==--- forEach for general containers -------------------------------------==//

/// Imlementation of forEach for containers, this overload is only enabled if
/// the container is not a Tuple. Benchmarking shows that this is faster than
/// both manually unrolling a the functor over a loop, and std::for_each.
/// 
/// \param[in]  container   The container to apply forEach to.
/// \param[in]  callable    The callable to call on the container.
/// \param[in]  args        The arguments for the callable.
/// \tparam     Container   The type of the container.
/// \tparam     Callable    The type of the callable.
/// \tparam     Args        The type of the arguments.
template <typename Container,
          typename Functor  ,
          typename... Args  ,
          typename Enable = std::enable_if_t<!isTuple<Container>, int>>
VoxxDeviceHost inline void
forEach(Container&& container, Functor&& functor, Args&&... args) {
  constexpr std::size_t unrollLevel = 4;
  std::size_t i = 0;
  if (container.size() >= unrollLevel) {
    for (; i < container.size() - unrollLevel; i += unrollLevel) {
      [[maybe_unused]] Unroll<unrollLevel> u(
        [i] (auto amount, auto&& cont, auto&& func, auto&&... as) { 
          func(cont[i + amount], as...);
        }, std::forward<Container>(container),
           std::forward<Functor>(functor)    ,
           std::forward<Args>(args)...       
      );
    }
  }

  // Whatever is left over:
  for (; i < container.size(); ++i)
    functor(container[i], std::forward<Args>(args)...);
}

} // namespace Voxx