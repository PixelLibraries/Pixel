//==--- Src/Thread/Thread.cpp ------------------------------ -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  Thread.cpp
/// \brief This file implements thread functionality.
//
//==------------------------------------------------------------------------==//

#include <Voxel/SystemInfo/CpuInfo.hpp>
#include <Voxel/Thread/Thread.hpp>
#include <Voxel/Utility/Debug.hpp>

#if defined(__APPLE__)
# include <sys/sysctl.h>
#elif defined(__linux__)
# include <sys/sysctl.h>
# include <sched.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace Voxx   {
namespace Thread {

//==--- ThreadException ----------------------------------------------------==//

const char* ThreadException::message() const noexcept {
  std::string threadNum = std::to_string(ThreadNumber), msg("");
  switch (ErrorType) {
    case Type::AffinitySetFailure:
      msg = "Failed to set affinity for thread: ";
      return (msg + threadNum).c_str();
      break;
    case Type::AffinityGetFailure:
      msg = "Failed to get afinity for unset thread: ";
      return (msg + threadNum).c_str();
      break;
    default:
      return "Unknown thread realted error";
  }
}

//==--- Helpers ------------------------------------------------------------==//

/// This anonymous namespace provides helper functionality.
namespace {

/// The function binds the execution context to the thread with \p threadId
/// on the currently executing cpu package.
/// It returns true if no errors ocurred.
/// \param[in]  threadId  The index of the thread to bind the context for.
bool bindContext(std::size_t threadId) {
#if defined(__linux__)
  cpu_set_t currentCpu;
  // Clear the cpu set and then add the cpu for threadId to the set.
  CPU_ZERO(&currentCpu);
  CPU_SET(threadId, &currentCpu);
  if (auto error = sched_setaffinity(0, sizeof(currentCpu), &currentCpu))
    throw ThreadException{ThreadException::Type::AffinityGetFailure, threadId};
#else

#endif
  return true;
}

} // namespace anon

//==--- Functions ----------------------------------------------------------==//

// Sets the affinity of the currently executing thread so that it executes
// exclusively on the cpu given by \p coreNumber. The core numbers should be
// user defined. An example useage may be to bind each thread to it's own core:
// \param[in]  coreNumber  The number of the core to set the thread to run on.
void setAffinity(std::size_t coreNumber) noexcept {
  bindContext(coreNumber);

  /// This will only throw in debug, so we mark
  /// the function noexcept for release mode.
  Debug::catcher([=] {
    const auto physicalCores = System::CpuInfo::cores();

    // If the system information has not been generated, then physicalCores will
    // be zero, so we don't throw an error.
    if (physicalCores && coreNumber >= physicalCores)
      throw DebugException{"Setting thread affinity past physical core count"};
  });
}

// Gets the affinity for the currently executing thread.
// If ``setAffinity(threadIndex)`` has been called, this will return a mask
// with __only__ ``threadIndex`` set.
// Returns a bitmask representing which logical threads the currently executing
// thread may run on.
BitMask getAffinity() noexcept {
  BitMask affinity{System::CpuInfo::threads()};
#if defined(__linux__)
  cpu_set_t availableMask;
  sched_getaffinity(0, sizeof(availableMask), &availableMask);
  for (std::size_t i = 0; i < affinity.size(); ++i) {
    if (CPU_ISSET(i, &availableMask) == 0)
      continue;
    affinity[i] = true;
  }
#else
#endif
  return affinity;
}

}} // namespace Voxx::Thread

#if defined(VoxxCpuArchArm)

// Implements the non-inlined function call which can act as a memory barrier.
void fakeMemoryBarrier() noexcept {}

#endif // VoxxCpuArchArm