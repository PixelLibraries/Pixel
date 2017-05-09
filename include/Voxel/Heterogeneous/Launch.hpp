//==--- Voxel/Heterogeneous/Launch.hpp --------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  Launch.hpp
/// \brief This file defines a class to launch callable classes implemented on
///        the host (Functors, Lambdas ...) on the device.
// 
//==------------------------------------------------------------------------==//

#pragma once

#include "Detail/LaunchImpl.hpp"

namespace Voxx   {
namespace Hetero {

/// Launches a device kernel which invokes the \p callable object.
/// \param[in]  gridDim       The dimensions of the grid.
/// \param[in]  blockDim      The dimensions of the blocks.
/// \param[in]  callable      The callable object to invoke.
/// \param[in]  args          The arguments for the callable.
/// \tparam     DimType       The type of the dimension specifiers.
/// \tparam     CallableType  The type of the callable object.
/// \tparam     Args          The types of the arguments.
template <typename DimType, typename CallableType, typename... Args>
void launch(DimType&&      gridDim ,
            DimType&&      blockDim,
            CallableType&& callable,
            Args&&...      args    ) {
    using Launcher = Detail::DeviceLauncher<CallableType, Args...>;
    Launcher::launch(gridDim                             ,
                     blockDim                            ,
                     std::forward<CallableType>(callable), 
                     std::forward<Args>(args)...         );
}

}} // namespace Voxx::Hetero