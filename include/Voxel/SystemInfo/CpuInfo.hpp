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

/// This namespace includes helper functionality for System related components.
namespace Detail {

/// The CpuidRegisters class wrap the eax, ebx, ecx, edx registers.
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

/// The TopologyMask struct defines a set of masks which define the system
/// processor topology.
struct TopologyMasks {
  unsigned smtMask;
  unsigned packageMask;
  unsigned coreMask;
  unsigned packageMaskShift;
  unsigned smtMaskWidth;
};

/// Wrapper around the cpudid function which is cross-platform, and which
/// returns the filled registers.
/// \param[in] invocationId The id of the function to call for cpuid.
inline CpuIdRegisters 
cpuid(unsigned invocationId, unsigned subfunction = 0) noexcept {
  CpuIdRegisters registers;
#if defined(_WIN32)
  __cpuid(static_cast<int*>(registers.data()), static_cast<int>(invocationId));
#else
  // For cpuid function 4, ecx is zero:
  asm volatile (
    "cpuid" 
      : "=a" (registers.values[0]), "=b" (registers.values[1]),
        "=c" (registers.values[2]), "=d" (registers.values[3])
      : "a"  (invocationId), "c" (subfunction));
#endif
  return registers;
}

} // namespace Detail

/// The CpuInfo struct provides information relating to the cpu. The data is
/// stored statically, per thread. The data can be 'refreshed' through the
/// refresh method:
/// 
/// ~~~cpp
/// CpuInfo::refresh()
/// ~~~
/// 
/// which will refresh the data __for only that thread__. In the single threaded
/// case this is insignificant. In the multi-threaded case, this is designed to
/// be used in a thread pool, where the data would be refreshed when the pool is
/// created.
/// 
/// If definitions are fed into the application by the build script, then the
/// constexpr versions can be used to get the information at compile time.
/// 
/// ...
struct CpuInfo {
 private:
  /// Forwarding reference constructor -- deleted by making it private to
  /// disable the construction of the class. __Note__ here that the modern,
  /// > c++11 ``delete`` does no work here, as the class is an aggregate class,
  /// hence we use deletion by private.
  /// \tparam Ts The types of any potential arguments.
  template <typename... Ts> CpuInfo(Ts&&...) {};

 public:
  /// Refreshes the static data from cpuid.
  static void refresh() noexcept {
    using namespace Detail;
    BasicFeatures    = cpuid(CpuidFunction::Features);
    CacheInfo        = cpuid(CpuidFunction::CacheCapabilities);
    ProcTopology     = cpuid(CpuidFunction::ProcessorTopology);
    ExtendedFeatures = cpuid(CpuidFunction::AdditionalFeatures);

    generateTopologyInfo();
  }

  /// Returns true if hyperthreading is supported by the cpu.
  static bool hyperthreading() noexcept {
    return extract(BasicFeatures.edx(), FeatureBitsEdx::Hyperthreading);
  }

  /// Returns the number of processors supported by the system.
  static bool processorCount() {
    return ProcessorCount;
  }

  /// Returns true if the cpu supports MMX intrinsics.
  static bool mmx() noexcept {
    return extract(BasicFeatures.edx(), FeatureBitsEdx::Mmx);
  }

  /// Returns true if the cpu supports MMX intrinsics.
  static bool aes() noexcept {
    return extract(BasicFeatures.ecx(), FeatureBitsEcx::Aes);
  }

  /// Returns true if the cpu supports SSE intrinsics.
  static bool sse() noexcept {
    return extract(BasicFeatures.edx(), FeatureBitsEdx::Sse);
  }

  /// Returns true if the cpu supports SSE2 intrinsics.
  static bool sse2() noexcept {
    return extract(BasicFeatures.edx(), FeatureBitsEdx::Sse2);
  }

  /// Returns true if the cpu supports SSE3 intrinsics.
  static bool sse3() noexcept {
    return extract(BasicFeatures.ecx(), FeatureBitsEcx::Sse3);
  }

  /// Returns true if the cpu supports SSSE3 intrinsics.
  static bool ssse3() noexcept {
    return extract(BasicFeatures.ecx(), FeatureBitsEcx::Ssse3);
  }

  /// Returns true if the cpu supports SSE4.1 intrinsics.
  static bool sse41() noexcept {
    return extract(BasicFeatures.ecx(), FeatureBitsEcx::Sse41);
  }

  /// Returns true if the cpu supports SSE4.2 intrinsics.
  static bool sse42() noexcept {
    return extract(BasicFeatures.ecx(), FeatureBitsEcx::Sse42);
  }

  /// Returns true if the cpu supports AVX intrinsics.
  static bool avx() noexcept {
    return extract(BasicFeatures.ecx(), FeatureBitsEcx::Avx);
  }

  /// Returns true if the cpu supports AVX2 intrinsics.
  static bool avx2() noexcept {
    return extract(ExtendedFeatures.ebx(), ExtendedFeatureBitsEbx::Avx2);
  }

  /// Returns true if the cpu supports sse2 intrinsics.
  static bool avx512() noexcept {
    return extract(BasicFeatures.ebx(), ExtendedFeatureBitsEbx::Avx512F);
  }

  /// Displays the information for the cpu.
  static void display();

 private:
  /// Register data holding basic cpu feature information.
  static thread_local Detail::CpuIdRegisters BasicFeatures;
  /// Register data holding cache information.
  static thread_local Detail::CpuIdRegisters CacheInfo;
  /// Register data holding processor and cache topology.
  static thread_local Detail::CpuIdRegisters ProcTopology;
  /// Register data holding extended features.
  static thread_local Detail::CpuIdRegisters ExtendedFeatures;
  /// Defines the masks which can be used to get the processor topology.
  static thread_local Detail::TopologyMasks  TopologyData;
  /// Defines the number of processors in the system.
  static thread_local int                    ProcessorCount;

  /// The CpuidFunction enum defines the values of the functions to return
  /// specific results with cpuid.
  enum CpuidFunction : uint32_t {
    /// Max leaf index supported by the processor.
    MaxLeaf            = 0x00000000,
    /// Basic features supported by the cpu.
    Features           = 0x00000001,
    /// Cache and TLB information.
    CacheCapabilities  = 0x00000002,
    /// Topology of the processor cores.
    ProcessorTopology  = 0x00000004,
    /// Additional features supported by the cpu.
    AdditionalFeatures = 0x00000007,
    /// Checks if Leaf B is supported.
    LeafB              = 0xB
  };

  /// The FeatureBitsEcx enum defines which bits in the edx register represent
  /// which features.
  enum FeatureBitsEcx : uint8_t {
    Sse3       =  0,   //!< SSE2 instructions.
    Ssse3      =  9,   //!< Suplemental SSE3 instructions.
    Fma        = 12,   //!< Fuse multiply add.
    Sse41      = 19,   //!< SSE4.1 instructions.
    Sse42      = 20,   //!< SSE4.2 instructions.
    Aes        = 25,   //!< Aes instructions.
    Avx        = 28,   //!< AVX instructions.
    RandGen    = 29,   //!< On chip random number generator.
    Hypervisor = 30    //!< Running on a hypervisor.
  };

  /// This FeatureBitsEdx enum defines which bits in the ecx register
  /// represent which features.
  enum FeatureBitsEdx : uint8_t {
    TimstampCounter  =  4,    //!< Time stamp counter.
    Mmx              = 23,    //!< Multimedia instructions.
    Sse              = 25,    //!< SSE instructions.
    Sse2             = 26,    //!< SSE2 instructions.
    SelfSnoop        = 27,    //!< Cpu supports self snoop.
    Hyperthreading   = 28,    //!< Hyperthreading support.
  };

  /// The ExtendedFeatureBitEbx enum defines which bits in the ebx register
  /// represent which extended features.
  enum ExtendedFeatureBitsEbx : uint8_t {
    Avx2       =  5,  //!< AVX2 instructions.
    Avx512F    = 16,  //!< AVX512 foundation instructions.
    Avx512Dq   = 17,  //!< AVX512 double and quadword instructions.
    RandSeed   = 18,  //!< If RDSEED instruction is available.
    Avx512Ifma = 21,  //!< AVX512 integer fuse multiple add instructions.
    Avx512Pf   = 26,  //!< AVX512 prefetch instructions.
    Avx512Er   = 27,  //!< AVX512 exponential and reciprocol instructions.
    Avx512Cd   = 28,  //!< AVX512 conflict detection instructions.
    Avx512Bw   = 30,  //!< AVX512 byte and word instructions.
    Avx512Vl   = 31   //!< AVX512 vector length instructions 
  };

  /// The ExtendedFeatureBitEcx enum defines which bits in the ecx register
  /// represent which extended features.
  enum ExtendedFeatureBitsEcx : uint8_t {
    Avx512Vbmi =  1,  //!< AVX512 vector bit manipulation instructions.
    ProcId     = 22,  //!< Read processor id instruction.
  };

  /// The ExtendedFeatureBitEdx enum defines which bits in the edx register
  /// represent which extended features.
  enum ExtendedFeatureBitsEdx : uint8_t {
    Avx512Nn    = 2,  //!< AVX512 neural network instructions.
    Avx512Fmaps = 3,  //!< AVX512 multiply accumulation single precision.
  }; 

  /// Extracts a proprety from register \p reg by shifting the register by \p
  /// shift amount and extracting the least significant bit.
  /// \param[in] reg    The register to get a proprety from.
  /// \param[in] shift  The amount to shift \p reg by, i.e the bit index of the
  ///                   property.
  static bool extract(uint32_t reg, uint8_t shift) noexcept {
    return (reg >> shift) & 0x01;
  }

  /// Gets the bits between \p start and \p end, inclusive, and returns the
  /// result.
  static uint32_t getBits(uint32_t data, uint8_t start, uint8_t end) {
    return (data >> start) & ((1 << end) - 1); 
  }

  /// Sets up the topology data structures.
  static void createTopologyStructures();

  /// Determines the system core and cache topology.
  static void generateTopologyInfo();

  /// Gets constants if the processor supports leaf B.
  static void getConstantsLeafB();

  /// Gets constants for legacy mode (processor doesn't support leaf B).
  static void getConstantsLegacy();

  /// Sets the number of processors in the system.
  static void findProcessorCount();

};

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