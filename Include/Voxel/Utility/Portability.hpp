//==--- Voxel/Utility/Portability.hpp ---------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  Portability.hpp
/// \brief This file defines macros to allow correct compilation for host and
///        device code.
//
//==------------------------------------------------------------------------==//

#pragma once

#if defined(__CUDA_ARCH__)

// Cuda device code compilation:
#define VoxxCudaDeviceCompile

#if defined(__clang__) && !defined(__CUDA__)

/// Definition for when clang is compiling CUDA code in host mode.
# define VoxxClangCudaHostCompile

#elif defined(__clang__) && defined(__CUDA__)

/// Definition for when clang is compiling CUDA code in device mode.
# define VoxxClangCudaDeviceCompile

#endif // clang and cuda
#endif // cuda_arch

#if defined(VoxxCudaSupported)
// System has CUDA:

#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__) || defined(__clang__)  // GPU Environment, or clang:
# define VoxxHost       __host__
# define VoxxDevice     __device__
# define VoxxDeviceHost __host__ __device__
#else
# define VoxxHost
# define VoxxDevice
# define VoxxDeviceHost
#endif  // __CUDACC__ || __clang__

#else
// System doesn't have cuda

# define VoxxHost
# define VoxxDevice
# define VoxxDeviceHost

#endif // VoxxCudaSupported