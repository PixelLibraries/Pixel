//==--- Voxel/Utility/Debug.hpp ---------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel 
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  Debug.hpp
/// \brief This file defines debugging functionality.
//
//==------------------------------------------------------------------------==//

#pragma once

#include "Portability.hpp"
#include <exception>
#include <string>

// Include relevant headers:
#if !defined(NDEBUG)
#include <cassert>
#include <cstdio>
#include <iostream>
#endif

#if defined(NDEBUG)
// Release mode definitions

/// Macro to assert a condition.
#define VoxxAssert(condition, message)

#if !defined(VoxxCudaSupported)
// If cuda is not supported, remove all error checks.
# define VoxxCudaErrorCheck(result)
#else
// If cuda is supported, just evaluate the expression.
# define VoxxCudaErrorCheck(result) (result)
#endif // VoxxCudaErrorCheck

#else
// Debug mode definitions:

/// Macro to check for Cuda call errors, when in debug mode.
#define VoxxCudaErrorCheck(result) {                                          \
  Voxx::checkCudaError((result), __FILE__, __LINE__); }

/// Macro to assert a condition.
#define VoxxAssert(condition, message)                                        \
  assert(condition && message)

namespace Voxx {

/// Checks if a cude error code was a success, and if not, then it prints
/// the error message.
/// \param[in] errCode The cuda error code.
/// \param[in] file    The file where the error was detected.
/// \param[in] line    The line in the file where the error was detected.
inline void checkCudaError(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess)  {
    printf("\nTerninated (Cuda Error):");
    printf("\n\tError : %s\n\tFile  : %s\n\tLine  : %-7i\n\n",
           cudaGetErrorString(code), file, line);
    std::terminate();
   }
}

} // namespace Voxx

#endif // NDEBUG
      
//==--- General debug ------------------------------------------------------==//

namespace Voxx {

/// The Debug exception class can be used when throwing exceptions from
/// the debugCheck() function.
class DebugException : public std::exception {
  std::string Message;  //!< The message to display.

 public:
  /// Constructor -- initializes the message.
  /// \param[in]  message   The message to display.
  DebugException(std::string message) : Message(message) {}

  /// Displays the message.
  virtual const char* what() const noexcept override final {
    const auto msg = std::string("\nDebugException:\n\t") + Message + "\n";
    return msg.c_str();
  }
};

namespace Debug {

/// Invokes the functor if NDEBUG is not defined and catches all exceptions,
/// otherwise (if NDEBUG __is__ defined__ removes the call to the functor,
/// as well as the catch code, which will allow to compiler to remove this
/// code entirely for release builds.
/// 
/// This will only catch exceptions derived from std::exception, which should be
/// all exceptions.
/// 
/// \param[in]  functor The functor to execute in debug mode.
/// \param[in]  args    The arguments to the functor.
/// \tparam     Functor The type of the functor.
/// \tparam     Args    The types of the arguments.
template <typename Functor, typename... Args>
inline void catcher(Functor&& functor, Args&&... args) {
#if !defined(NDEBUG)
  try {
    functor(std::forward<Args>(args)...);
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
  }
#endif
}

}} // namespace Voxx::Debug