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

#include <Voxel/Bit/BitManip.hpp>
#include <Voxel/Bit/BitMask.hpp>

#if defined(_WIN32)
# include <limits.h>
# include <intrin.h>
#else
#include <stdint.h>
#endif

namespace Voxx   {
namespace System {

/// \todo change this to use the correct values by feeding in the value at
/// compile time.
static constexpr std::size_t destructiveInterfaceSize() noexcept {
  return 64;
}

/// The CpuidRegisters struct wrap the eax, ebx, ecx, edx registers.
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
};

/// The TopologyMask struct defines a set of masks which define the system
/// processor topology.
struct TopologyMasks {
  unsigned threadMask;          //!< A mask for getting the thread number.
  unsigned packageMask;         //!< A mask for getting the package number.
  unsigned coreMask;            //!< A mask for getting the core number.
  unsigned packageMaskShift;    //!< Amount to shift for the package.
  unsigned threadMaskWidth;     //!< Amount to shift for the threads.
  
  /// Creates the topology masks.
  static TopologyMasks create();
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

/// This namespace defines a simpler interface for common cpuid calls.
namespace CpuId {

/// The Functoin enum defines the values of the functions to return
/// specific results with cpuid.
enum Function : uint32_t {
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

//==--- Functions ----------------------------------------------------------==//

/// Returns the APIC ID for the currently bound core/thread.
static inline uint32_t apicid() noexcept {
  auto regs = cpuid(Function::MaxLeaf);
  // If leaf b is supported, then we can return x2APPIC ID:
  if (regs.eax() >= Function::LeafB) 
    return cpuid(Function::LeafB).edx();

  // Otherwise we need to return a zero extended 8 bit initial id:
  return getBits(cpuid(Function::Features).ebx(), 24, 31);  
}

/// Returns the max leaf supported for a call to cpuid.
static inline uint32_t maxleaf() noexcept {
  return cpuid(Function::MaxLeaf).eax();
}

/// Returns true of the cpu has support for leaf B.
static inline bool supportsLeafB() noexcept {
  return cpuid(Function::LeafB).ebx() != 0;
}

/// Returns the processor topology raw data.
/// \param[in]  leaf  The leaf to get the information for.
static inline CpuIdRegisters processorTopology(uint32_t leaf = 0) noexcept {
  return cpuid(Function::ProcessorTopology, leaf);
}

/// Returns register data from which features can be extracted.
static inline CpuIdRegisters features() noexcept {
  return cpuid(Function::Features);
}

/// Returns register data from which extended features can be extracted.
static inline CpuIdRegisters extendedFeatures() noexcept {
  return cpuid(Function::AdditionalFeatures);
}

} // namespace Cpuid

/// The ThreadInfo struct defines infromation for a thread. This is simply a
/// class which stores the APIC ID of the thread and provides functions to
/// extract the components from the APIC ID.
class ThreadInfo {
 public:
  /// Default constructor -- sets the APIC ID to 0.
  constexpr ThreadInfo() noexcept : Apicid(0) {}

  /// Constructor -- sets the APIC ID using the given APIC ID.
  /// \param[in]  apicid  The APIC ID of the thread.
  constexpr ThreadInfo(uint32_t apicid) noexcept
  : Apicid(apicid) {}

  /// Returns the package to which the thread belongs.
  /// \param[in]  masks The topology masks to use to extract the parameters.
  constexpr uint32_t package(const TopologyMasks& masks) const noexcept {
    return (Apicid & masks.packageMask) >> masks.packageMaskShift;
  }

  /// Returns the core number to which the thread belongs.
  /// \param[in]  masks The topology masks to use to extract the parameters.
  constexpr uint32_t core(const TopologyMasks& masks) const noexcept {
    return (Apicid & masks.coreMask) >> masks.threadMaskWidth;
  }

  /// Returns the number of the thread in the core.
  /// \param[in]  masks The topology masks to use to extract the parameters.
  constexpr uint32_t thread(const TopologyMasks& masks) const noexcept {
    return Apicid & masks.threadMask;
  }

  /// Prints a summary of the thread information.
  void print(const TopologyMasks& masks) const;

 private: 
  uint32_t Apicid;  //!< The apicid for the thread.
};

/// Defines information related to the cache.
struct CacheInfo {
  /// The Type enum defines the type a cache can have.
  enum class Type : uint8_t {
    Invalid     = 0x00,   //!< Cache type is invalid.
    Data        = 0x01,   //!< Cache is a data cache.
    Instruction = 0x02,   //!< Cache is an instruction cache.
    Unified     = 0x03    //!< Cache is both data and instruction.
  };

  /// Default constructor -- intiailizes all fields to be zero.
  constexpr CacheInfo() noexcept
  : MainProperties(0), LineProperties(0), Sets(0) {}

  /// Constuctor -- creates the cache information from the register values
  /// returned by a call to cpuid(CpuidFunction::ProcessorTopology, subleaf)
  /// where the subleaf is the leaf to get the information for,
  /// \param[in] regs   The registers to use to get the cache information.
  constexpr CacheInfo(CpuIdRegisters regs) noexcept
  : MainProperties(regs.eax()), LineProperties(regs.ebx()), Sets(regs.ecx()) {}

  /// Returns the asociativity of the cache.
  constexpr uint32_t associativity() const noexcept {
    return getBits(LineProperties, 22, 31) + 1;
  }

  /// Returns the number of partiions (blocks) in the cache.
  constexpr uint32_t partitions() const noexcept {
    return getBits(LineProperties, 12, 21) + 1;
  }

  /// Returns the size of a cache line, in bytes.
  constexpr uint32_t lineSize() const noexcept {
    return getBits(LineProperties, 0, 11) + 1;
  }

  /// Returns the number of sets in the cache.
  constexpr uint32_t sets() const noexcept {
    return Sets + 1;
  }

  /// Returns the size, in bytes, of the cache.
  constexpr std::size_t size() const noexcept {
    return lineSize() * partitions() * associativity() * sets();
  }

  /// Returns the level of the cache.
  constexpr uint8_t level() const noexcept {
    return getBits(MainProperties, 5, 7);
  }

  /// Returns the maximum number of threads sharing the cache.
  constexpr uint32_t maxSharing() const noexcept {
    return getBits(MainProperties, 14, 25) + 1;
  }

  /// Returns the core and package information.
  constexpr uint32_t coresPerPackage() const noexcept {
    return getBits(MainProperties, 26, 31) + 1;
  }

  /// Returns the type of the cache.
  constexpr Type type() const noexcept {
    return static_cast<Type>(getBits(MainProperties, 0, 4));
  }

  /// Returns true if the cache is a data cache, i.e it is either a data cache
  /// or a unified cache.
  constexpr bool isDataCache() const noexcept {
    return type() == Type::Data || type() == Type::Unified;
  }

  /// Converts the type of the cache to its name value.
  std::string typeAsName() const;

  /// Prints the cache information.
  void print() const;

  uint32_t MainProperties;  //!< General properties of the cache.
  uint32_t LineProperties;  //!< Properties of a cache line.
  uint32_t Sets;            //!< The number of sets in the cache.
};

/// TheCpuInfo struct defines the number of packages (or sockets), cores
/// (the number of hardware cores), and logical cores (the number of threads) in
/// the system.
struct CpuInfo {
  /// Alias for the type of container used to store thread information.
  using ThreadInfoContainer  = std::vector<ThreadInfo>;
  /// Alias for the type of container used to store cache information.
  using CacheInfoContainer   = std::vector<CacheInfo>;
  /// Alias for the type of sharing information container.
  using SharingInfoContainer = std::vector<uint32_t>;

  /// Refreshes the cpu information. This can be used in situations where the
  /// system may have changed (i.e a server where more nodes have become
  ///  available).
  static void refresh();

  //==--- Properties -------------------------------------------------------==//


  /// Returns the number of physical cores.
  static uint32_t cores() noexcept {
    return PhysicalCores;
  }

  /// Returns the number of threads per core.
  static uint32_t threadsPerCore() noexcept {
    return threads() / cores();
  }

  /// Returns the number of logical threads for the cpu.
  static uint32_t threads() noexcept {
    return Threads.size();
  }

  /// Returns the number of bytes in a cache line.
  static uint32_t cacheLineSize() noexcept {
    return Caches[0].lineSize();
  }

  /// Returns the information for a level of the cache.
  /// \param[in] level The level to get the cache info for (1 = L1, 2 = L2, 
  //                   3 = L3 ...)
  static CacheInfo cacheInfo(std::size_t level = 1) {
    return Caches[level - 1];
  }

  //==--- Features ---------------------------------------------------------==//

  /// Returns true if hyperthreading is supported by the cpu.
  static bool hyperthreading() noexcept {
    return getBit(BasicFeatures.edx(), FeatureBitsEdx::Hyperthreading);
  }

  /// Returns true if the cpu supports MMX intrinsics.
  static bool mmx() noexcept {
    return getBit(BasicFeatures.edx(), FeatureBitsEdx::Mmx);
  }

  /// Returns true if the cpu supports MMX intrinsics.
  static bool aes() noexcept {
    return getBit(BasicFeatures.ecx(), FeatureBitsEcx::Aes);
  }

  /// Returns true if the cpu supports SSE intrinsics.
  static bool sse() noexcept {
    return getBit(BasicFeatures.edx(), FeatureBitsEdx::Sse);
  }

  /// Returns true if the cpu supports SSE2 intrinsics.
  static bool sse2() noexcept {
    return getBit(BasicFeatures.edx(), FeatureBitsEdx::Sse2);
  }

  /// Returns true if the cpu supports SSE3 intrinsics.
  static bool sse3() noexcept {
    return getBit(BasicFeatures.ecx(), FeatureBitsEcx::Sse3);
  }

  /// Returns true if the cpu supports SSSE3 intrinsics.
  static bool ssse3() noexcept {
    return getBit(BasicFeatures.ecx(), FeatureBitsEcx::Ssse3);
  }

  /// Returns true if the cpu supports SSE4.1 intrinsics.
  static bool sse41() noexcept {
    return getBit(BasicFeatures.ecx(), FeatureBitsEcx::Sse41);
  }

  /// Returns true if the cpu supports SSE4.2 intrinsics.
  static bool sse42() noexcept {
    return getBit(BasicFeatures.ecx(), FeatureBitsEcx::Sse42);
  }

  /// Returns true if the cpu supports AVX intrinsics.
  static bool avx() noexcept {
    return getBit(BasicFeatures.ecx(), FeatureBitsEcx::Avx);
  }

  /// Returns true if the cpu supports AVX2 intrinsics.
  static bool avx2() noexcept {
    return getBit(ExtendedFeatures.ebx(), ExtendedFeatureBitsEbx::Avx2);
  }

  /// Returns true if the cpu supports sse2 intrinsics.
  static bool avx512() noexcept {
    return getBit(ExtendedFeatures.ebx(), ExtendedFeatureBitsEbx::Avx512F);
  }

  //==--- Utilities --------------------------------------------------------==//
 
  /// Returns the number of __cores__ (not threads) shared by the \p level th
  /// level of cache which is shared by __cores__. Where \p level is __0__ the
  /// number of cores sharing the first level of cache which has core sharing
  /// is returned.
  /// 
  /// I.e if L1 and L2 are only shared by threads on a core, then L3 is the
  /// first level of core sharing, and using \p level ``= 0`` will return the
  /// number of cores which share L3.
  /// 
  /// \param[in] level  The level for which to get the number of shared cores
  ///                   for.
  static uint32_t sharedCores(uint32_t level = 0);

  /// Prints all the cpu information.
  static void print();

 private:
  /// Forwarding reference constructor -- deleted by making it private to
  /// disable the construction of the class. __Note__ here that the modern,
  /// > c++11 ``delete`` does no work here, as the class is an aggregate class,
  /// hence we use deletion by private.
  /// \tparam Ts The types of any potential arguments.
  template <typename... Ts> CpuInfo(Ts&&...) {};

  /// Properties for each of the system threads.
  static thread_local ThreadInfoContainer  Threads;
  /// Properties for each of the levels of cache.
  static thread_local CacheInfoContainer   Caches;
  /// Number of threads which share each level of cache.
  static thread_local SharingInfoContainer SharingSizes;
  /// Register data holding basic cpu feature information.
  static thread_local CpuIdRegisters       BasicFeatures;
  /// Register data holding extended cpu feature information.
  static thread_local CpuIdRegisters       ExtendedFeatures;
  /// Masks used to getermine the topology of the system processors and cache.
  static thread_local TopologyMasks        Masks;
  /// The number of CPU packages in the system.
  static thread_local std::size_t          Packages;
  /// The number of physical cores in the system.
  static std::size_t          PhysicalCores;
  /// The index of the first level of cache shared by physical cpus.
  static thread_local std::size_t          CoreSharingLevel;

  /// Sets the number of threads in the system.
  static void getThreadCount();

  /// Gets the number of physical cores. This must only be called one the number
  /// of threads is known.
  static void getCoreCount();

  /// Gets a mask for the threads which need to be parsed.
  static BitMask getAffinityMask();

  /// Gets the cache information.
  static void getCacheInfo();

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
};

}} // namespace Voxx::System