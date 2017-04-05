//==--- Utility/Debug.hpp ---------------------------------- -*- C++ -*- ---==//
//            
//                                Voxel : Utility 
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

#ifndef VOXEL_UTILITY_DEBUG_HPP
#define VOXEL_UTILITY_DEBUG_HPP

#include "Portability.hpp"

#ifdef NDEBUG
// Release mode definitions

/// Macro to check for Cuda call errors.
#define VoxxCudaErrorCheck(result)
/// Macro to assert a condition.
#define VoxxAssert(condition, message)

#else
// Debug mode definitions:
#include <cassert>
#include <exception>
#include <iostream>

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
    std::cout << "\nTerminated (Cuda Error): " << cudaGetErrorString(code)
              << "\n\t file : "                << file
              << "\n\t line : "                << line << "\n\n";
    std::terminate();
   }
}

} // namespace Voxx

#endif // NDEBUG
#endif // VOXEL_UTILITY_DEBUG_HPP