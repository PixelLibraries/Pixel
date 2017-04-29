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
# include <sys/sysctl.h>
# include "CpuInfoApple.cpp"
#elif defined(__linux)
# include <unistd.h>
# include <sys/sysctl.h>
# include "CpuInfoLinux.cpp"
#elif defined(_WIN32)
# include <windows.h>
# include "CpuInfoWindows.cpp"
#else
# error Unsupported platform!
#endif

namespace Voxx    {
namespace System  {

using namespace Detail;

// Initialize CpuProperties variables:
thread_local CpuIdRegisters
  CpuInfo::BasicFeatures    = cpuid(CpuInfo::CpuidFunction::Features);
thread_local CpuIdRegisters 
  CpuInfo::CacheInfo        = cpuid(CpuInfo::CpuidFunction::CacheCapabilities);
thread_local CpuIdRegisters
  CpuInfo::ProcTopology     = cpuid(CpuInfo::CpuidFunction::ProcessorTopology);
thread_local CpuIdRegisters
  CpuInfo::ExtendedFeatures = cpuid(CpuInfo::CpuidFunction::AdditionalFeatures);
thread_local int CpuInfo::ProcessorCount = 1;
thread_local TopologyMasks CpuInfo::TopologyData{};

void CpuInfo::display() {
  using namespace Voxx::Io;
  using Out = Output<Mode::Console>;

  Out::banner("Cpu Info");
  Out::writeResult("Processor Count" , ProcessorCount   );
  Out::writeResult("MMX    Supported", CpuInfo::mmx()   );
  Out::writeResult("AES    Supported", CpuInfo::mmx()   );
  Out::writeResult("SSE    Supported", CpuInfo::sse()   );
  Out::writeResult("SSE2   Supported", CpuInfo::sse2()  );
  Out::writeResult("SSE3   Supported", CpuInfo::sse3()  );
  Out::writeResult("SSSE3  Supported", CpuInfo::ssse3() );
  Out::writeResult("SSE41  Supported", CpuInfo::sse41() );
  Out::writeResult("SSE42  Supported", CpuInfo::sse42() );
  Out::writeResult("AVX    Supported", CpuInfo::avx()   );
  Out::writeResult("AVX2   Supported", CpuInfo::avx2()  );
  Out::writeResult("AVX512 Supported", CpuInfo::avx512());
  Out::banner();
}

void CpuInfo::findProcessorCount() {
  ProcessorCount = 1;
#if defined(_WIN32)
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  ProcessorCount = sysinfo.dwNumberOfProcessors;
#elif defined(__linux__) || (__APPLE__)
  // This version works only for osx >= 10.4
  ProcessorCount = sysconf(_SC_NPROCESSORS_ONLN);
#else
  int mib[4];
  std::size_t length = sizeof(ProcessorCount);

  mib[0] = CTL_HW;
  mib[1] = HW_AVAILCPU;
  sysctl(mib, 2, &ProcessorCount, &length, nullptr, 0);
  // If the above didn't work, try the other flag:
  if (ProcessorCount < 1) {
    mib[1] = HW_NCPU;
    sysctl(mib, 2, &ProcessorCount, &length, nullptr, 0);

    // We bail out here =(
    if (ProcessorCount < 1)
      ProcessorCount = 1;
  }
#endif
}

void CpuInfo::generateTopologyInfo() {
  using namespace Detail;
  auto regs     = cpuid(CpuidFunction::MaxLeaf);
  auto maxLeaf  = regs.eax();
  bool hasLeafB = false;

  // Check if we have support for leaf B:
  if (maxLeaf >= CpuidFunction::LeafB)
    hasLeafB = (cpuid(CpuidFunction::LeafB).ebx() != 0);

  // Multi core systems have hyperthreading:
  if (hyperthreading()) { 
    if (hasLeafB) 
      getConstantsLeafB();
    else 
      getConstantsLegacy();
  } else {  // If no hyperthreading, then there is only a single core:
    // ...
  }
}

void CpuInfo::createTopologyStructures() {
  findProcessorCount();
}

void CpuInfo::getConstantsLeafB() {
  // This enum defines the two types of levels we can get:
  enum LevelType { Smt = 1, Core = 2 };
  // These bools set whether or not a core/thread was found.
  bool coreReported = false, threadReported = false;
  // These determine the leaf, level, and shift amounts.
  int  subLeaf = 0, levelType = 0, levelShift     = 0;
  // This defines a mask for the physical and logical cores.
  uint64_t corePlusSmtMask = 0;

  while (true) {
    auto regs = cpuid(CpuidFunction::LeafB, subLeaf);

    // Check that the subleaf is valid:
    if (!regs.ebx())
      break;

    levelType  = getBits(regs.ecx(), 8, 15);
    levelShift = getBits(regs.eax(), 0, 4 );

    switch (levelType) {
      // For the SMT level, the levelShift
      // defines the SMT_Mask_Width:
      case LevelType::Smt:
        TopologyData.smtMask      = ~((-1) << levelShift);
        TopologyData.smtMaskWidth = levelShift; 
        threadReported            = true;
        break;

      // For the Core level, the levelShift
      // defines the CorePlusSMT_Mask_width:
      case LevelType::Core:
        corePlusSmtMask               = ~((-1) << levelShift);
        TopologyData.packageMask      = (-1) ^ corePlusSmtMask;
        TopologyData.packageMaskShift = levelShift;
        coreReported                  = true;
        break;

      default: break;
    }
    ++subLeaf;
  }

  if (threadReported && coreReported) {
    TopologyData.coreMask = corePlusSmtMask ^ TopologyData.smtMask;
  } else if (threadReported && !coreReported) {
    TopologyData.coreMask         = 0;
    TopologyData.packageMask      = (-1) ^ TopologyData.smtMask;
    TopologyData.packageMaskShift = TopologyData.smtMaskWidth;
  } else {
    // This should never happen, we can throw an error here ...
  }
}

void CpuInfo::getConstantsLegacy() {
  std::cout << "Legacy\n";
}

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