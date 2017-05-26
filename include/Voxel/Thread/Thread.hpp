//==--- Voxel/Thread/Thread.hpp ---------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  Thread.hpp
/// \brief This file defines thread related functionality.
//
//==------------------------------------------------------------------------==//

#pragma once

#include "Detail/ThreadImpl.hpp"
#include <Voxel/Bit/BitMask.hpp>
#include <exception>
#include <string>

namespace Voxx   {
namespace Thread {

/// The ThreadException class overrides the standard exception class to define
/// thread related exceptions.
class ThreadException : public std::exception {
 public:
  /// The Type enum defines the types of thread realted exceptions.
  enum class Type : uint8_t {
    AffinitySetFailure  = 0x00, //!< Failure when setting the affinity.
    AffinityGetFailure  = 0x01, //!< Failure when getting the affinity.
    Oversubscription    = 0x02  //!< More threads than cores.
  };

 public:
  /// Constructor -- initializes the type of the thread error.
  /// \param[in]  type          The type of the thread error.
  /// \param[in]  threadNumber  The number of the thread with the error.
  ThreadException(Type type, std::size_t threadNumber) noexcept
  : ErrorType(type), ThreadNumber(threadNumber) {}

  /// Display a message for the exception.
  virtual const char* what() const noexcept override {
    const std::string msg = std::string("\nThread Exception:\n\t") + message();
    return msg.c_str();
  }

 private:
  Type        ErrorType;    //!< The type of the thread exception.
  std::size_t ThreadNumber; //!< The number of the thread with the error.
                  
  /// Returns char string pointer with an appropriate message for the type of
  /// the error.
  const char* message() const noexcept;
};

/// Sets the affinity of the currently executing thread so that it executes
/// exclusively on the cpu given by \p coreNumber. The core numbers should be
/// user defined. An example useage may be to bind each thread to it's own core:
/// 
/// ~~~cpp
/// void run(int coreNumber) {
///   Voxx::Thread::setAffinity(coreNumber);
///   
///   // Do work ..
/// }
///
/// int main(int argc, char** argv) {
///   static constexpr std::size_t threadCount = 8;
///   std::vector<std::thread> threads(threadCount);
///   
///   for (int i = 0; i < threads.size(); ++i)
///     threads[i] = std::thread(run, i);
///
///   for (auto& thread : threads)
///     thread.join();
/// }
/// ~~~
/// 
/// \param[in]  coreNumber  The number of the core to set the thread to run on.
void setAffinity(std::size_t coreNumber) noexcept;

/// Gets the affinity for the currently executing thread.
/// If ``setAffinity(threadIndex)`` has been called, this will return a mask
/// with __only__ ``threadIndex`` set.
/// Returns a bitmask representing which logical threads the currently executing
/// thread may run on.
BitMask getAffinity() noexcept;

/// Emits a memeory barrier to prevent reordering of instructions. On systems
/// with a strongly ordered memory model this is simply a compiler barrier,
/// hoever, on systems with a weakly ordered memory model this will emit the
/// appropriate instruction to ensure memory orderring.
inline void memoryBarrier() noexcept {
  VoxxMemoryBarrier();
}

}} // namespace Voxx::Thread
