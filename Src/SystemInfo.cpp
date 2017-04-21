//==--- Src//SystemInfo.cpp -------------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  SystemInfo.cpp
/// \brief This file provides an implementation of system related information
///        functionality.
//
//==------------------------------------------------------------------------==//

#include <Voxel/SystemInfo/SystemInfo.hpp>
#include <Voxel/SystemInfo/CpuInfo.hpp>
#include <Voxel/Io/Io.hpp>

// Include the relevant implementation:
#if defined(__APPLE__)
# include "CpuInfoApple.cpp"
#elif defined(WIN32)
# include "CpuInfoWindows.cpp"
#elif defined(linux)
# include "CpuInfoLinux.cpp"
#else
# error Unsupported platform!
#endif

namespace Voxx    {
namespace System  {

std::string intrinsicAsString(IntrinsicSet intrinsic) {
  switch (intrinsic) {
    case IntrinsicSet::Avx2 : return "AVX 2.0"  ;
    case IntrinsicSet::Avx1 : return "AVX 1.0"  ;
    case IntrinsicSet::Sse42: return "SSE 4.2"  ;
    case IntrinsicSet::Sse41: return "SSE 4.1"  ;
    case IntrinsicSet::Ssse3: return "SSSE 3.0" ;
    case IntrinsicSet::Sse3 : return "SSE 3.0"  ;
    case IntrinsicSet::Sse2 : return "SSE 2.0"  ;
    case IntrinsicSet::Sse  : return "SSE 1.0"  ;
    case IntrinsicSet::Neon : return "Neon"     ;
    default:                  return "Invalid"  ;
  }  
}

void writeCpuInfo() {
    using namespace Voxx::Io;
    using      Out       = Output<Mode::Console>;
    const auto intrinsic = intrinsicAsString(intrinsicSet());

    Out::banner("Cpu Info");
    Out::writeResult("Physical CPUs"     , cpuCount()                  );
    Out::writeResult("Physical Cores"    , physicalCores()             );
    Out::writeResult("Logical Cores"     , logicalCores()              );
    Out::writeResult("Physical Cores/CPU", physicalCores() / cpuCount());
    Out::writeResult("Logical Cores/CPU" , logicalCores()  / cpuCount());
    Out::writeResult("Intrinsic Support" , intrinsic                   );
    Out::writeResult("Cacheline Size (B)", cachelineSize()             );
    Out::writeResult("L1 Cache Size (B)" , l1CacheSize()               );
    Out::writeResult("L2 Cache Size (B)" , l2CacheSize()               );
    Out::writeResult("L3 Cache Size (B)" , l3CacheSize()               );
    Out::writeResult("L1 Sharing (Cores)", l1Sharing()                 );
    Out::writeResult("L2 Sharing (Cores)", l2Sharing()                 );
    Out::banner();   
}

// TODO: Add support for output method.
void writeSystemInfo() {
  writeCpuInfo();
}

}} // namespace Voxx::System