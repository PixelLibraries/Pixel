//==--- Src/CpuInfoLinux.cpp ------------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  CpuInfoLinux.cpp
/// \brief This file implements the CPU information functionality for linux.
//
//==------------------------------------------------------------------------==//

#if defined(__linux__)

#include <Voxel/Io/Io.hpp>
#include <Voxel/SystemInfo/CpuInfo.hpp>
#include <cstdlib>

namespace Voxx   {
namespace System {

namespace {

// The CachePropert enum defines the offset into the array of cache properties
// returned when "hw.cacheconfig" is queried from sysctlbyname.
enum CacheProperty : uint8_t {
  L1Sharing = 0x01,   //!< The offset of the L1 sharing property.
  L2Sharing = 0x02    //!< The offset of the L2 sharing property.
};

} // namespace annon

std::size_t cpuCount() {
  return 0;
}

std::size_t physicalCores() {
  return 0;
}

std::size_t logicalCores() {
  return 0;
}

std::size_t cachelineSize() {
  return 0;
}

std::size_t l1CacheSize() {
  return 0;
}

std::size_t l2CacheSize() {
  return 0;
}

std::size_t l3CacheSize() {
  return 0;
}

std::size_t l1Sharing() {
  return 0;
}

std::size_t l2Sharing() {
  return 0;
}

IntrinsicSet intrinsicSet() {
  return IntrinsicSet::Invalid;
}

}} // namespace Voxx::System

#endif // __linux__
