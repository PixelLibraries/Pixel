//==--- Voxel/Function/CallableFwd.hpp --------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  CallableFwd.hpp
/// \brief This file forward declared the available callable implementations.
//
//==------------------------------------------------------------------------==//

#pragma once

namespace Voxx     {
namespace Function {

/// This class defines functionality for wrapping a callable object and its
/// arguments. It's the callable which should be used in most cases, unless the
/// arguments need to be dynamically allocated, in which case the
/// DynamicCallable should be used.
/// 
/// \tparam   CallableType  The type of the callable object being wrapped.
/// \tparam   Args          The types of the callable's arguments.
template <typename CallableType, typename... Args>
class Callable;

/// This class defines functionality for wrapping a callable object which
/// dynamically allocates the arguments
/// arguments which may change (i.e they may be modified on the device, and are
/// non-const lvalue reference types and hence we need to get the modified
/// versions back from the device onto the host).
/// 
/// \tparam   CallableType  The callable object to invoke.
/// \tparam   Args          The types of the arguments.
template <typename CallableType, typename... Args>
class DynamicCallable;


}} // namespace Voxx::Function