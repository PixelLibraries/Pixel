//==--- Src/SystemInfo/CpuInfoApple.cpp -------------------- -*- C++ -*- ---==//
//            
//                                Voxel : Utility 
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  CpuInfoApple.cpp
/// \brief This file implements the CPU information functionality for apple.
//
//==------------------------------------------------------------------------==//

#if defined(__APPLE__)

#include <Voxel/Utility/SystemInfo/CpuInfo.hpp>
#include <Voxel/Utility/Io/Io.hpp>
#include <sys/sysctl.h>
#include <cstdlib>

namespace Voxx    {
namespace Utility {
namespace System  {

namespace {

// The CachePropert enum defines the offset into the array of cache properties
// returned when "hw.cacheconfig" is queried from sysctlbyname.
enum CacheProperty : uint8_t {
  L1Sharing = 0x01,   //!< The offset of the L1 sharing property.
  L2Sharing = 0x02    //!< The offset of the L2 sharing property.
};

/// Returns the value of the property at offset \p index in the property given
/// by \p name. This overload should be called for properties which return
/// an array of 64 bit values.
/// 
/// \note This overload is more expensive because the array needs to be
/// allocated and freed.
/// 
/// \param[in]  name    The name of the propery to get.
/// \param[in]  index   The index of the property in the array to get.
std::size_t getProperty(const char* name, std::size_t index) {
  char* buffer = nullptr; std::size_t size = 0, value = 0;
  sysctlbyname(name, nullptr, &size, nullptr, 0);
  buffer = static_cast<char*>(malloc(size));
  sysctlbyname(name, buffer, &size, nullptr, 0);
  value = *reinterpret_cast<std::size_t*>(buffer + sizeof(std::size_t) * index);
  free(buffer);
  return value;
}

/// Returns the value of the property given by \p name. This overload should be
/// called for properties which return a single value.
/// \param[in]  name  The name of the propery to get.
std::size_t getProperty(const char* name) {
  std::size_t value = 0, size = sizeof(std::size_t);
  sysctlbyname(name, &value, &size, 0, 0);
  return value;
}

} // namespace annon

std::size_t cpuCount() {
  return getProperty("hw.packages");
}

std::size_t physicalCores() {
  return getProperty("hw.physicalcpu");
}

std::size_t logicalCores() {
  return getProperty("hw.logicalcpu");
}

std::size_t cachelineSize() {
  return getProperty("hw.cachelinesize");
}

std::size_t l1CacheSize() {
  return getProperty("hw.l1dcachesize");
}

std::size_t l2CacheSize() {
  return getProperty("hw.l2cachesize");
}

std::size_t l3CacheSize() {
  return getProperty("hw.l3cachesize");
}

std::size_t l1Sharing() {
  return getProperty("hw.cacheconfig", CacheProperty::L1Sharing);
}

std::size_t l2Sharing() {
  return getProperty("hw.cacheconfig", CacheProperty::L2Sharing);
}

IntrinsicSet intrinsicSet() {
  const auto queries = { 
    std::make_pair("hw.optional.avx2_0"          , IntrinsicSet::Avx2 ),
    std::make_pair("hw.optional.avx1_0"          , IntrinsicSet::Avx1 ),
    std::make_pair("hw.optional.sse4_2"          , IntrinsicSet::Sse42),
    std::make_pair("hw.optional.sse4_1"          , IntrinsicSet::Sse41),
    std::make_pair("hw.optional.supplementalsse3", IntrinsicSet::Ssse3),
    std::make_pair("hw.optional.sse3"            , IntrinsicSet::Sse3 ),
    std::make_pair("hw.optional.sse2"            , IntrinsicSet::Sse2 ),
    std::make_pair("hw.optional.sse"             , IntrinsicSet::Sse  )
  };
  for (const auto& query : queries)
    if (getProperty(query.first))
      return query.second;
  return IntrinsicSet::Invalid;
}

}}} // namespace Voxx::Utility;:System

#endif // __APPLE__