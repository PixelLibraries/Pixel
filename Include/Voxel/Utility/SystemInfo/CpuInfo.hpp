//==--- Utility/SystemInfo/CpuInfo.hpp --------------------- -*- C++ -*- ---==//
//            
//                                Voxel : Utility 
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

#include <Voxel/Utility/Bitwise.hpp>
#include <Voxel/Utility/Algorithm/Algorithm.hpp>
#include <cstdio>
#include <string>

#if   defined(__APPLE__)
# include <sys/sysctl.h>
#elif defined(WIN32)
#elif defined(linux)
#else
# error Unsupported Platform
#endif

namespace Voxx    { 
namespace Utility {

/// Defines vector instruction support for the CPU.
enum CpuIntrinsic : uint8_t {
  Avx2    = 0x00,
  Avx1    = 0x01,
  Sse42   = 0x02,
  Sse41   = 0x03,
  Ssse3   = 0x04,
  Sse3    = 0x05,
  Sse2    = 0x06,
  Sse     = 0x07,
  Neon    = 0x08,
  Invalid = 0x09
};

/// Gets the a struct representing the name and version of the intrinsics.
static auto 
printableIntrinsicInfo(uint8_t intrinsic) -> std::pair<const char*, float> {
  switch (intrinsic) {
    case CpuIntrinsic::Avx2 : return std::make_pair("Avx"    , 2.0f);
    case CpuIntrinsic::Avx1 : return std::make_pair("Avx"    , 1.0f);
    case CpuIntrinsic::Sse42: return std::make_pair("Sse"    , 4.2f);
    case CpuIntrinsic::Sse41: return std::make_pair("Sse"    , 4.1f);
    case CpuIntrinsic::Ssse3: return std::make_pair("Ssse"   , 3.0f);
    case CpuIntrinsic::Sse3 : return std::make_pair("Sse"    , 3.0f);
    case CpuIntrinsic::Sse2 : return std::make_pair("Sse"    , 2.0f);
    case CpuIntrinsic::Sse  : return std::make_pair("Sse"    , 1.0f);
    case CpuIntrinsic::Neon : return std::make_pair("Neon"   , 1.0f);
    default:                  return std::make_pair("Invalid", 0.0f);
  }
}

/// This struct stores CPU information, and can be used both at runtime and
/// compile time (if a binary representation of the information is known at
/// compile time. The class defines the following information related to the
/// CPU:
/// 
/// intrinsic       : The latest supported intrinsic set.
/// physical cores  : The number of physical cores for the CPU.
/// logical core    : The number of logical cores for the CPU.
/// cacheline size  : The number of bytes in the cache line.
/// L1 cache size   : The size of the L1 cache.
/// L2 cache size   : The size of the L2 cache, in bytes.
/// 
/// It's assume here that a single CPU does not have more than 32 physical
/// cores, and no more than 64 logical cores.
class CpuInfo {
 public:
  //==--- Con/destructors --------------------------------------------------==//
  
  /// Default constructor -- creates uninitialized information.
  CpuInfo() : Value(0) {
    fillProperties();
  }

  /// Constructor -- sets the binary value of the cpu properties.
  constexpr CpuInfo(uint64_t value) noexcept : Value(value) {}

  //==--- Getters ----------------------------------------------------------==//
  
  /// Returns the latest supported intrinsic set.
  constexpr auto intrinsic() const noexcept -> uint64_t {
    return getProperty(Mask::Intrinsic);
  }

  /// Returns the number of physical cores.
  constexpr auto physicalCores() const noexcept -> uint64_t {
    return getProperty(Mask::PhysicalCores);
  }

  /// Returns the number of logical cores.
  constexpr auto logicalCores() const noexcept -> uint64_t {
    return getProperty(Mask::LogicalCores);
  }

  /// Returns the size of a cache line, in bytes.
  constexpr auto cachelineSize() const noexcept -> uint64_t {
    return getProperty(Mask::CachelineSize);
  }

  /// Returns the size of the L1 cache, in bytes.
  constexpr auto cacheSizeL1() const noexcept -> uint64_t {
    return getProperty(Mask::CacheSizeL1);
  }

  /// Returns the size of the L1 cache, in bytes.
  constexpr auto cacheSizeL2() const noexcept -> uint64_t {
    return getProperty(Mask::CacheSizeL2);
  }

  //==--- Setters ----------------------------------------------------------==//

  /// Sets the supported set of intrinsics.
  /// \param[in]  intrinsic   The value of the supported intrinsics
  constexpr void setIntrinsics(uint64_t intrinsic) noexcept {
    setProperty(intrinsic, Mask::Intrinsic);
  }

  /// Sets the number of physical cores.
  /// \param[in]  cores   The number of physical cores.
  constexpr void setPhysicalCores(uint64_t cores) noexcept {
    setProperty(cores, Mask::PhysicalCores);
  }

  /// Sets the number of logical cores.
  /// \param[in]  cores   The number of logical cores.
  constexpr void setLogicalCores(uint64_t cores) noexcept {
    setProperty(cores, Mask::LogicalCores);
  }

  /// Set the size of the cacheline, in bytes.
  /// \param[in]  size  The size of the cache line, in bytes.
  constexpr void setCachelineSize(uint64_t size) noexcept {
    setProperty(size, Mask::CachelineSize);
  }

  /// Sets the size of the level 1 cache, in bytes.
  /// \param[in]  size  The size of the first level cache, in bytes.
  constexpr void setL1CacheSize(uint64_t size) noexcept {
    setProperty(size >> magnitudeShift, Mask::CacheSizeL1);
  }

  /// Sets the size of the level 2 cache, in B.
  /// \param[in]  size  The size of the second level cache, in bytes.
  constexpr void setL2CacheSize(uint64_t size) noexcept {
    setProperty(size >> magnitudeShift, Mask::CacheSizeL2);
  }

  /// Prints the CPU information.
  void print() const {
    constexpr auto format    = "| %-28s%-10s%38llu |\n";
    constexpr auto hexFormat = "| %-28s%-10s%#38lx |\n";
    constexpr auto intFormat = "| %-28s%-10s%34s %2.1f |\n";
    const     auto intrin    = printableIntrinsicInfo(intrinsic());
    const     auto banner    = [] () { 
      printf("|==%s==|\n", std::string(74, '-').c_str()); 
    };

    banner();
    printf(format   , "Cpu Number"        , ":", uint64_t{0}                );
    printf(intFormat, "Intrinsic Set"     , ":", intrin.first, intrin.second);
    printf(format   , "Physical Cores"    , ":", physicalCores()            );
    printf(format   , "Logical Cores"     , ":", logicalCores()             );
    printf(format   , "Cacheline Size (B)", ":", cachelineSize()            );
    printf(format   , "L1 Cache Size (B)" , ":", cacheSizeL1()              );
    printf(format   , "L2 Cache Size (B)" , ":", cacheSizeL2()              );
    printf(hexFormat, "Raw Representation", ":", Value                      );
    banner();
  }

 private:
  std::size_t Value; //!< The value which defines the supported functionality.
  
  /// Defines the number of shifts to change 3 orders of magnitude 
  /// (1 << 10 = 1024).
  static constexpr uint64_t magnitudeShift = 10;
                  
  /// The Mask enum defines the bit masks for the fields.
  enum Mask : uint64_t {
    Intrinsic     = 0x00000000000001F, //!< Intrinsic set  : Bits [00 - 04].
    PhysicalCores = 0x0000000000003E0, //!< Physical cores : Bits [05 - 09].
    LogicalCores  = 0x00000000000FC00, //!< Logical cores  : Bits [10 - 15].
    CachelineSize = 0x000000001FF0000, //!< Size in Bytes  : Bits [16 - 24].
    CacheSizeL1   = 0x00001FFFE000000, //!< Size in KiB    : Bits [25 - 40].
    CacheSizeL2   = 0xFFFFE0000000000, //!< Size in Kib    : Bits [41 - 59].
  };

  /// Sets the bits for a property, first clearing the property, and then
  /// setting it.
  /// \param[in] value The value to set the property to.
  /// \param[in] mask  The mask for the property.
  constexpr void setProperty(uint64_t value, uint64_t mask) noexcept {
    Value = (Value & (~mask)) | ((value << firstSetBitIndex(mask)) & mask);
  }

  /// Gets one of the properties.
  /// \param[in] mask  The mask for the property.
  constexpr auto getProperty(uint64_t mask) const noexcept -> uint64_t {
    return (Value & mask) >> firstSetBitIndex(mask);
  }

  //==--- Non-constexpr platform-specific helpers --------------------------==//

#if defined(__APPLE__)

  /// Returns the highest supported set of intrinsics.
  static inline auto getCpuIntrinsics() -> uint8_t {
    auto queries = { 
      std::make_pair("hw.optional.avx2_0",           CpuIntrinsic::Avx2 ),
      std::make_pair("hw.optional.avx1_0",           CpuIntrinsic::Avx1 ),
      std::make_pair("hw.optional.sse4_2",           CpuIntrinsic::Sse42),
      std::make_pair("hw.optional.sse4_1",           CpuIntrinsic::Sse41),
      std::make_pair("hw.optional.supplementalsse3", CpuIntrinsic::Ssse3),
      std::make_pair("hw.optional.sse3"            , CpuIntrinsic::Sse3 ),
      std::make_pair("hw.optional.sse2"            , CpuIntrinsic::Sse2 ),
      std::make_pair("hw.optional.sse"             , CpuIntrinsic::Sse  )
    };

    for (const auto& query : queries) {
      std::size_t value = 0, size = sizeof(std::size_t);
      sysctlbyname(query.first, &value, &size, 0, 0);
      if (value) return query.second;
    }
    return CpuIntrinsic::Invalid;
  }

  /// Fills all the cpu properties.
  void fillProperties() {
    setIntrinsics(getCpuIntrinsics());

    // Sets the property calling setFunction with value.
    auto setProperty = [this] (auto& setFunction, auto value) {
      (this->*setFunction)(value);
    };
    auto properties  = std::make_tuple(
      std::make_pair("hw.cachelinesize", &CpuInfo::setCachelineSize),
      std::make_pair("hw.physicalcpu"  , &CpuInfo::setPhysicalCores),
      std::make_pair("hw.logicalcpu"   , &CpuInfo::setLogicalCores ),
      std::make_pair("hw.l1dcachesize" , &CpuInfo::setL1CacheSize  ),
      std::make_pair("hw.l2cachesize"  , &CpuInfo::setL2CacheSize  )
    );

    forEach(properties, [setProperty] (auto&& property) {
      std::size_t value = 0, size = sizeof(std::size_t);
      sysctlbyname(property.first, &value, &size, 0, 0);
      setProperty(property.second, value);
    });
  }

#elif defined(WIN32)

#elif defined(linux)

#endif
};

}} // namespace Voxx::Utility