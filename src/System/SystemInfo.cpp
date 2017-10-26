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
#include <Voxel/Thread/Thread.hpp>
#include <Voxel/Io/Io.hpp>
#include <algorithm>
#include <random>

// Include the relevant implementation:
#if defined(__APPLE__)
# include <sys/sysctl.h>
#elif defined(__linux)
# include <unistd.h>
# include <sys/sysctl.h>
# include <sched.h>
#elif defined(_WIN32)
# include <windows.h>
#else
# error Unsupported platform!
#endif

namespace Voxx    {
namespace System  {

// Initialize all the static CpuInfo:
thread_local CpuInfo::ThreadInfoContainer  CpuInfo::Threads(0);
thread_local CpuInfo::CacheInfoContainer   CpuInfo::Caches(0);
thread_local CpuInfo::SharingInfoContainer CpuInfo::SharingSizes(0);
thread_local CpuIdRegisters                CpuInfo::BasicFeatures;
thread_local CpuIdRegisters                CpuInfo::ExtendedFeatures;
thread_local TopologyMasks                 CpuInfo::Masks;
thread_local std::size_t                   CpuInfo::Packages(0);
std::size_t                                CpuInfo::PhysicalCores(0);
thread_local std::size_t                   CpuInfo::CoreSharingLevel(0);  

//==--- TopologyMasks ------------------------------------------------------==//

namespace {

TopologyMasks getConstantsLeafB() {
  // This enum defines the two types of levels we can get:
  enum LevelType { Thread = 1, Core = 2 };
  // These bools set whether or not a core/thread was found.
  bool coreReported = false, threadReported = false;
  // These determine the leaf, level, and shift amounts.
  int subLeaf = 0, levelType = 0, levelShift = 0;
  // This defines a mask for the physical and logical cores.
  uint64_t corePlusSmtMask = 0;

  TopologyMasks masks;
  while (true) {
    auto regs = cpuid(CpuId::Function::LeafB, subLeaf);

    // Check that the subleaf is valid:
    if (!regs.ebx())
      break;

    levelType  = getBits(regs.ecx(), 8, 15);
    levelShift = getBits(regs.eax(), 0, 4 );

    switch (levelType) {
      // For the SMT level, the levelShift
      // defines the SMT_Mask_Width:
      case LevelType::Thread:
        masks.threadMask      = ~((-1) << levelShift);
        masks.threadMaskWidth = levelShift; 
        threadReported        = true;
        break;

      // For the Core level, the levelShift
      // defines the CorePlusSMT_Mask_width:
      case LevelType::Core:
        corePlusSmtMask        = ~((-1) << levelShift);
        masks.packageMask      = (-1) ^ corePlusSmtMask;
        masks.packageMaskShift = levelShift;
        coreReported           = true;
        break;

      default: break;
    }
    ++subLeaf;
  }

  if (threadReported && coreReported) {
    masks.coreMask = corePlusSmtMask ^ masks.threadMask;
  } else if (threadReported && !coreReported) {
    masks.coreMask         = 0;
    masks.packageMask      = (-1) ^ masks.threadMask;
    masks.packageMaskShift = masks.threadMaskWidth;
  } else {
    // This should never happen, we can throw an error here ...
  }
  return masks;
}

TopologyMasks getConstantsLegacy() {
  return TopologyMasks{};
}

} // namespace anon

/// Creates a set of masks which can be used to get the package, core and thread
/// numbers for an executing thread.
TopologyMasks TopologyMasks::create() {
  bool          hasLeafB = false;
  TopologyMasks masks;

  if (CpuId::maxleaf() >= CpuId::Function::LeafB)
    hasLeafB = CpuId::supportsLeafB();

  // Multi core systems have hyperthreading:
  if (CpuInfo::hyperthreading()) { 
    if (hasLeafB) 
      masks = getConstantsLeafB();
    else 
      masks = getConstantsLegacy();
  } else {  // If no hyperthreading, then there is only a single core:
    // ...
  }
  return masks;
}

//==--- Cache --------------------------------------------------------------==//

std::string CacheInfo::typeAsName() const {
  switch (type()) {
    case Type::Invalid     : return "Invalid"     ;
    case Type::Data        : return "Data"        ;
    case Type::Instruction : return "Instruction" ;
    case Type::Unified     : return "Unified"     ;
    default                : return "Error"       ;
  }
}

//==--- CpuInfo ------------------------------------------------------------==//

void CpuInfo::refresh() {
  BasicFeatures    = CpuId::features();
  ExtendedFeatures = CpuId::extendedFeatures();
  Masks            = TopologyMasks::create();
  getThreadCount();
  getCoreCount();
  getCacheInfo();
}

void CpuInfo::getThreadCount() {
  std::size_t threadCount = 0;
#if defined(_WIN32)
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  threadCount = sysinfo.dwNumberOfProcessors;
#elif defined(__linux__) || (__APPLE__)
  // This version works only for osx >= 10.4
  // Gets the number of online processors.
  threadCount = sysconf(_SC_NPROCESSORS_ONLN);
#else
  int mib[4];
  std::size_t length = sizeof(threadCount);

  mib[0] = CTL_HW;
  mib[1] = HW_AVAILCPU;
  sysctl(mib, 2, &threadCount, &length, nullptr, 0);
  // If the above didn't work, try the other flag:
  if (logicalCores < 1) {
    mib[1] = HW_NCPU;
    sysctl(mib, 2, &threadCount, &length, nullptr, 0);

    // We bail out here =(
    if (threadCount < 1)
      threadCount = 1;
  }
#endif
  Threads.resize(threadCount);
}

void CpuInfo::getCacheInfo() {
  // We can be pretty sure that there are not 256 levels of cache:
  constexpr uint32_t maxLeaves = 256;

  // Loop through the leaves until we reach an
  // invald cache, at which point we are done:
  for (uint32_t subleaf = 0; subleaf <  maxLeaves; ++subleaf) {
    CacheInfo cache(CpuId::processorTopology(subleaf));
    if (cache.type() == CacheInfo::Type::Invalid)
      break;

    // We don't care about the sharing for instruction only caches.
    if (cache.isDataCache()) {
      const auto level = cache.level();
      SharingSizes.resize(level);
      SharingSizes[level - 1] = cache.maxSharing();
      if (!CoreSharingLevel && (SharingSizes[level - 1] > threadsPerCore()))
        CoreSharingLevel = level - 1;
    }
    Caches.push_back(std::move(cache));
  }

  // Ensure that the caches are ordered by level:
  std::sort(Caches.begin(), Caches.end(),
    [] (const auto& first, const auto& second) {
      return first.level() < second.level();
    }
  );
}

void CpuInfo::getCoreCount() {
  auto affinity = Thread::getAffinity();
  auto coreList = std::vector<uint32_t>(0);

  for (std::size_t i = 0; i < Threads.size(); ++i) {
    // Check if logical processor i is valid (note that our BitMask
    // implementation is contiguous, while the OS specific ones are not). If the
    // logical processor is valid, we bind it's context, and then derive
    // parameters for the core.
    if (affinity[i]) {
      Thread::setAffinity(i);
      Threads[i] = ThreadInfo{CpuId::apicid()};

      const auto coreNumber   = Threads[i].core(Masks);
      const auto coreIterator = std::find(coreList.begin(),
                                          coreList.end()  ,
                                          coreNumber      );
      if (coreIterator == std::end(coreList))
        coreList.push_back(coreNumber);
    }
  }
  PhysicalCores = coreList.size();
}

uint32_t CpuInfo::sharedCores(uint32_t level) {
  const auto levelIndex = (CoreSharingLevel + level >= SharingSizes.size())
                        ? CoreSharingLevel : CoreSharingLevel + level;
  return std::min(cores(), SharingSizes[levelIndex]);
}

//==--- Printing functions -------------------------------------------------==//

void ThreadInfo::print(const TopologyMasks& masks) const {
  using namespace Voxx::Io;
  using Out = Output<Mode::Console>;

  Out::writeResult("APIC ID"       , Apicid        );
  Out::writeResult("Package Number", package(masks));
  Out::writeResult("Core    Number", core(masks)   );
  Out::writeResult("Thread  Number", thread(masks) );
}

void CacheInfo::print() const {
  using namespace Voxx::Io;
  using Out = Output<Mode::Console>;

  Out::writeResult("Type"              , typeAsName()                  );
  Out::writeResult("Level"             , static_cast<unsigned>(level()));
  Out::writeResult("Cores per Package" , coresPerPackage()             );
  Out::writeResult("Max Thread Sharing", maxSharing()                  );
  Out::writeResult("Size (kB)"         , size() >> 10                  );
  Out::writeResult("Line Size (B)"     , lineSize()                    );
  Out::writeResult("Partitions"        , partitions()                  );
  Out::writeResult("Associativity"     , associativity()               );
  Out::writeResult("Sets"              , sets()                        );
}

void CpuInfo::print() {
  using namespace Voxx::Io;
  using Out = Output<Mode::Console>;

  Out::banner("Cpu Features:");
  Out::writeResult("Cores"               , cores()         );
  Out::writeResult("Threads"             , threads()       );
  Out::writeResult("Threads Per Core"    , threadsPerCore());
  Out::writeResult("Hyperthreading"      , hyperthreading());
  Out::writeResult("MMX     Instructions", mmx()           );
  Out::writeResult("AES     Instructions", aes()           );
  Out::writeResult("SSE     Instructions", sse()           );
  Out::writeResult("SSE2    Instructions", sse2()          );
  Out::writeResult("SSE3    Instructions", sse3()          );
  Out::writeResult("SSSE3   Instructions", ssse3()         );
  Out::writeResult("SSE41   Instructions", sse41()         );
  Out::writeResult("SSE42   Instructions", sse42()         );
  Out::writeResult("AVX     Instructions", avx()           );
  Out::writeResult("AVX2    Instructions", avx2()          );
  Out::writeResult("AVX512F Instructions", avx512()        );

  Out::banner("Thread Information:");
  std::size_t i = 0;
  for (const auto& thread : Threads) {
    thread.print(Masks);
    if (++i < Threads.size())
      Out::banner();
  }
  Out::banner("Cache Information:");
  for (const auto& cache : Caches) {
    cache.print();
    Out::banner();
  }
}

}} // namespace Voxx::System