//==--- Voxel/SystemInfo/CpuInfo.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  CpuInfo.hpp
/// \brief This file defines functionality to get cpu information.
//
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Algorithm/Algorithm.hpp>
#include <Voxel/Io/IoFwd.hpp>
#include <Voxel/Utility/Bitwise.hpp>
#include <string>

namespace Voxx   {
namespace System {

/// Defines vector instruction support for the CPU.
enum class IntrinsicSet : uint8_t {
  Avx2    = 0x00,   //!< Intel AVX  2.0 instructions.
  Avx1    = 0x01,   //!< Intel AVX  1.0 instructions.
  Sse42   = 0x02,   //!< Intel SSE  4.2 instructions.
  Sse41   = 0x03,   //!< Intel SSE  4.1 instructions.
  Ssse3   = 0x04,   //!< Intel SSSE 3.0 instructions.
  Sse3    = 0x05,   //!< Intel SSE  3.0 instructions.
  Sse2    = 0x06,   //!< Intel SSE  2.0 instructions.
  Sse     = 0x07,   //!< Intel SSE  1.0 instructions.
  Neon    = 0x08,   //!< Arm Neon instructions.
  Invalid = 0x09    //!< Invalid intrinsic set.
};  

/// Returns the total number of CPUs in the system.
std::size_t cpuCount();

/// Returns the total number of physical cores in the system.
std::size_t physicalCores();

/// Returns the total number of logical cores in the system.
std::size_t logicalCores();

/// Returns the size of the cache line on the CPU, in bytes.
std::size_t cachelineSize();

/// Returns the size of the L1 cache.
std::size_t l1CacheSize();

/// Returns the size of the L2 cache, in bytes.
std::size_t l2CacheSize();

/// Returns the size of the L3 cache, in bytes.
std::size_t l3CacheSize();

/// Returns the number of logical cores which share the L1 cache.
std::size_t l1Sharing();

/// Returns the number of logical cores which share the L2 cache.
std::size_t l2Sharing();

/// Returns the highest supported set of intrinsics.
IntrinsicSet intrinsicSet();

/// Returns a string representation of the intrinsics.
/// \param[in]  intrinsicSet  The intrinsic set to get the string representation
///                           of.
std::string intrinsicAsString(IntrinsicSet intrinsicSet);

/// Writes the cpu information for the system.
/// \todo Add support for the type of output.
void writesCpuInfo();

/// This namespace contains constexpr versions of the functions, if they are
/// available. If the function is not available, 0 is returned. This allows
/// implementations to specialize functions for runtime and compile time, for
/// example:
/// 
/// \code{.cpp}
/// // Default version, cpuCores is known at compile time:
/// template <int cpuCores>
/// inline auto doSomethingImpl() {
///   // Imprmentation using compile time coreCount ...
/// }
/// 
/// // Runtime version, cpuCount = 0 therefore not known at compile time:
/// template <>
/// auto doSomething<System::Cx::UnknownCpuCount>() {
///   // Implementation using runtime version ...
/// }
/// 
/// // Wrapper function which calls the appropriate implementation:
/// auto doSomething() {
///   return doSomethingImpl<System::Cx::cpuCount()>();
/// }
/// \endcode
/// 
/// With c++17, ```if constexpr(System::Cx::cpuCount) {}``` could also be used.
namespace Cx {

/// Returns the total number of CPUs in the system.
static constexpr auto cpuCount() -> std::size_t {
#if defined(VoxxCpuCount)
  return VoxxCpuCount;
#else
  return 0;
#endif // VoxxCpuCount
}

/// Returns the number of physical cores in the system.
static constexpr auto physicalCores() -> std::size_t {
#if defined(VoxxPhysicalCores)
  return VoxxPhysicalCores;
#else
  return 0;
#endif
}

/// Returns the number of logical cores in the system.
static constexpr auto logicalCores() -> std::size_t {
#if defined(VoxxLogicalCores)
  return VoxxLogicalCores;
#else
  return 0;
#endif
}

/// Returns the size of a cacheline, in bytes.
static constexpr auto cachelineSize() -> std::size_t {
#if defined(VoxxCachelineSize)
  return VoxxCachelineSize;
#else
  return 0;
#endif
}

/// Returns the size of the L1 cache, in bytes.
static constexpr auto l1CacheSize() -> std::size_t {
#if defined(VoxxL1CacheSize)
  return VoxxL1CacheSize;
#else
  return 0;
#endif
}

/// Returns the size of the L2 cache, in bytes.
static constexpr auto l2CacheSize() -> std::size_t {
#if defined(VoxxL2CacheSize)
  return VoxxL2CacheSize;
#else
  return 0;
#endif
}

/// Returns the size of the L3 cache, in bytes.
static constexpr auto l3CacheSize() -> std::size_t {
#if defined(VoxxL2CacheSize)
  return VoxxL2CacheSize;
#else
  return 0;
#endif
}

/// Returns the number of logical processors which share the L1 cache.
static constexpr auto l1Sharing() -> std::size_t {
#if defined(VoxxL1Sharing)
  return VoxxL1Sharing;
#else
  return 0;
#endif
}

/// Returns the number of logical processors which share the L2 cache.
static constexpr auto l2Sharing() -> std::size_t {
#if defined(VoxxL2Sharing)
  return VoxxL2Sharing;
#else
  return 0;
#endif
}

/// Returns the supported intrinsic set.
static constexpr auto intrinsicSet() -> IntrinsicSet {
#if defined(VoxxIntrinsicSet)
  return static_cast<IntrinsicSet>(VoxxIntrinsicSet);
#else
  return IntrinsicSet::Invalid;
#endif
}

}}} // namespace Voxx::System::Cx