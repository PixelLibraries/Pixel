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
#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>

#if defined(_WIN32)
# include <limits.h>
# include <intrin.h>
#else
#include <stdint.h>
#endif

namespace Voxx    {
namespace System  {

/// The CpuInfo namespace defines functionality to get the features for the
/// cpu, given inputs from the cpuid function.
namespace CpuInfo {

/// This Detail namespace wraps specific cpuid calls.
namespace Detail {

/// The CpuidRegisters wrap the eax, ebx, ecx, edx registers.
struct CpuIdRegisters {
  uint32_t values[4]; //!< The values of the register data.

  /// Default constructor -- sets data to zero.
  constexpr CpuIdRegisters() : values{0} {}

  /// Fills the register data from 2 64 bits numbers.
  /// \param[in] eabx Data for a and b registers, stored [a, b].
  /// \param[in] ecdx Data for c and d regisresr, stores [c, d].
  constexpr CpuIdRegisters(uint64_t eabx, uint64_t ecdx)
  : values{static_cast<uint32_t>((eabx >> 32) & 0xFFFFFFFF),
           static_cast<uint32_t>(eabx         & 0xFFFFFFFF),
           static_cast<uint32_t>((ecdx >> 32) & 0xFFFFFFFF),
           static_cast<uint32_t>(ecdx         & 0xFFFFFFFF)} {}

  /// Fills the register data from 4 32 bits numbers.
  /// \param[in] eaxVal Data for eax register.
  /// \param[in] ebxVal Data for ebx register.
  /// \param[in] ecxVal Data for ecx register.
  /// \param[in] edxVal Data for edx register.
  constexpr CpuIdRegisters(
    uint32_t eaxVal, uint32_t ebxVal, uint32_t ecxVal, uint32_t edxVal)
  : values{eaxVal, ebxVal, ecxVal, edxVal} {}

  /// Returns a raw pointer to the register data.
  uint32_t* data() { return &values[0]; }

  /// Returns the data in the eax register.
  constexpr uint32_t eax() const { return values[0]; }
  /// Returns the data in the ebx register.
  constexpr uint32_t ebx() const { return values[1]; }
  /// Returns the data in the ecx register.
  constexpr uint32_t ecx() const { return values[2]; }
  /// Returns the data in the edx register.
  constexpr uint32_t edx() const { return values[3]; }
  
  /// Prints the raw data:
  void print() {
    std::cout << '\n' << "Eax" << " : 0x" << std::hex << eax() << '\n'
                      << "Ebx" << " : 0x" << std::hex << ebx() << '\n'
                      << "Ecx" << " : 0x" << std::hex << ecx() << '\n'
                      << "Edx" << " : 0x" << std::hex << edx() << '\n';
  }
};

/// Wrapper around the cpudid function which is cross-platform, and which
/// returns the filled registers.
/// \param[in] invocationId The id of the function to call for cpuid.
inline CpuIdRegisters cpuid(unsigned invocationId) noexcept {
  CpuIdRegisters registers;
#if defined(_WIN32)
  __cpuid(static_cast<int*>(registers.data()), static_cast<int>(invocationId));
#else
  // For cpuid function 4, ecx is zero:
  asm volatile (
    "cpuid" 
      : "=a" (registers.values[0]), "=b" (registers.values[1]),
        "=c" (registers.values[2]), "=d" (registers.values[3])
      : "a"  (invocationId), "c" (0));
#endif
  return registers;
}

struct CpuProperties {
 public:
  static CpuProperties create() {
    if (Created)
      throw std::logic_error{"Attempt to recreate cpuinfo"};
    Created = true;
    return CpuProperties{};
  }

  ~CpuProperties() { Created = false; }

  static inline bool mmx() noexcept {
    return property(InfoAndFeatures.edx(), 23);
  }

  static inline bool sse() noexcept {
    return property(InfoAndFeatures.edx(), 25);
  }

  static inline bool sse2() noexcept {
    return property(InfoAndFeatures.edx(), 26);
  }

 private:
  static thread_local CpuIdRegisters InfoAndFeatures;
  static thread_local CpuIdRegisters CacheAndTlb;
  static thread_local CpuIdRegisters ThreadAndCoreTopology;
  static thread_local CpuIdRegisters ExtendedFeatures;
  static thread_local bool           Created;

  static bool property(uint32_t reg, uint32_t shift) noexcept {
    return (reg >> shift) & 0x01;
  }
};

/// The CpuProperty enum defines the invocation ids of the cpuid functions for
/// querying different properties.
enum Property : uint8_t {
  BasicFeatures      = 0x01, //!< Processor info and feature bits.
  Cache              = 0x02, //!< Cache and TLB capabilities.
  Topology           = 0x04, //!< Cores and cache topology.
  AdditionalFeatures = 0x07, //!< Additional features.
};

} // namespace Detail

/// The CpuId struct wraps the cpuid assembly call to return specific cpu data
/// which can be passed by the static class functions, for example, to determine
/// if mmx instructions are supported, one could do:
/// 
/// ~~~cpp
/// auto mmx = CpuInfo::mmx(CpuInfo::CpuId::features())
/// Returns true of the cpu supports

} // namespace CpuInfo

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