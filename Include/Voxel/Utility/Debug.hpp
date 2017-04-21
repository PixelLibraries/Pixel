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
#include <cassert>
#include <exception>
#include <cstdio>

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
    printf("Terninated (Cuda Error):");
    printf("\n\tError : %s\n\tFile  : %s\n\tLine  :%-7i\n\n",
           cudaGetErrorString(code), file, line);
    std::terminate();
   }
}

} // namespace Voxx

#endif // NDEBUG