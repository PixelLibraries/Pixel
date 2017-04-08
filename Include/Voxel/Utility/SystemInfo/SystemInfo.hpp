//==--- Utility/SystemInfo/SystemInfo.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Voxel : Utility 
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  SystemInfo.hpp
/// \brief This file defines functionality to get system information.
//
//==------------------------------------------------------------------------==//

#pragma once

#include "CpuInfo.hpp"
#include "GpuInfo.hpp"
#include <Voxel/Utility/Debug.hpp>
#include <Voxel/Utility/Portability.hpp>

namespace Voxx    {
namespace Utility {

/// The SystemInfo struct defines properties of the system.
/// \todo Change this for all system CPUs and GPUs.
class SystemInfo {
 public:
  /// Constructor -- creates the system information.
  SystemInfo() : Cpus{}, Gpus{0} {} 
 
  /// Prints the system information.
  void print() const {
    // for (const auto& cpu : Cpus)
    Cpus.print();
    // For (const auto& gpu : Gpus)
    Gpus.print();
  }
 private:
  CpuInfo Cpus;
  GpuInfo Gpus;
};

}} // namespace Voxx::Utility