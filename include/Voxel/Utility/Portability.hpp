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
/// \brief This file defines macros for cross platform support.
//
//==------------------------------------------------------------------------==//

#pragma once

#if defined(_WIN32)                                 // Windows
  #define VoxxCompilerMSVC
  #if _XBOX_VER >= 200                              // XBox 360
    #define VoxxSystemTargetXbox360
    #define VoxxCpuArchPowerPC
  #else
    #if defined(_M_X64)                             // 32 bit
      #define VoxxCpuArchX64
    #elif defined(_M_IX86)                          // 64 bit
      #define VoxxCpuArchX86
    #else
      #error Unsupported cpu architecture!
    #endif                                          // End 32/64 bit
  #endif                                            // End Xbox360/other
#elif defined(__GNUC__)                             // GCC
  #if defined(__clang__)                            // Clang
    #define VoxxCompilerClang
  #else                                             // GCC
    #define VoxxCompilerGCC
  #endif
#endif                                              // End Windows / GCC / Clang

// Properties specific to clang and gcc
#if defined (__GNUC__)                              // GCC or Clang
  #if defined (__linux__)                           // Linux
    #define VoxxSystemTargetLinux
  #endif                                            // End linux
  #if defined(__APPLE__)                            // Apple
    #define VoxxSystemTargetApple
  #endif                                            // End Apple
  #if defined(__x86_64__) || defined(__i386__)      // X86
    #define VoxxCpuArchX86
  #elif defined(__arm__)
    // Arm V7
    #if defined(__ARM_ARCH_7__)   ||                                           \
        defined(__ARM_ARCH_7A__)  ||                                           \
        defined(__ARM_ARCH_7EM__) ||                                           \
        defined(__ARM_ARCH_7R__)  ||                                           \
        defined(__ARM_ARCH_7M__)  ||                                           \
        defined(__ARM_ARCH_7S__)
      #define VoxxCpuArchArm 7
    // Arm V6
    #elif defined(__ARM_ARCH_6__)   ||                                         \
          defined(__ARM_ARCH_6J__)  ||                                         \
          defined(__ARM_ARCH_6K__)  ||                                         \
          defined(__ARM_ARCH_6T2__) ||                                         \
          defined(__ARM_ARCH_6Z__)  ||                                         \
          defined(__ARM_ARCH_6ZK__)
      #define VoxxCpuArchArm 6
    #else
      #error Unsupported arm cpu version!
    #endif                                            // End ARM
  #else
    #error Unsupported cpu architecture!
  #endif                                              // End CPU arch
#endif                                                // End GCC || Clang

// CUDA properties
#define VoxxCudaNoCompilerKind 0
#define VoxxCudaNvccKind       1
#define VoxxCudaClangKind      2

// Check for a cuda capable compiler.
#if defined(__NVCC__)                                     // NVCC
  /// Cuda compiler is set to NVCC
  #define VoxxCudaCompiler VoxxCudaNvccKind
#elif defined(__clang__)                                  // Clang
  /// Cuda compiler is set to clang.
  #define VoxxCudaCompiler VoxxCudaClangKind
#else                                                     // Unsupported
  /// No supported cuda compiler.
  #define VoxxCudaCompiler VoxxCudaNoCompilerKind
#endif                                                    // End NVCC | Clang

// Definitions specifically when compiling on the device.
#if defined(__CUDA_ARCH__)                                // Cuda supported
  // Cuda device code compilation:
  #define VoxxCudaDeviceCompile

  #if defined(__clang__) && !defined(__CUDA__)            // Cuda and clang
    /// Definition for clang compiling cuda host code
    #define VoxxHostCompile
  #elif defined(__clang__) && defined(__CUDA__)
    /// Definition for clang compiling cuda device code
    #define VoxxDeviceCompile
  #elif defined(__NVCC__)
    /// Definition for nvcc compiling, only compiles on the device.
    #define VoxxDeviceCompile
  #endif                                                  // End clang and cuda
#endif                                                    // End cuda

// If the system has cuda:
#if defined(VoxxCudaSupported)
  #include <cuda.h>
  #include <cuda_runtime.h>

  #if defined(__CUDACC__) || defined(__clang__)  // GPU Environment, or clang:
    #define VoxxHost       __host__
    #define VoxxDevice     __device__
    #define VoxxDeviceHost __host__ __device__
  #else
    #define VoxxHost
    #define VoxxDevice
    #define VoxxDeviceHost
  #endif                                         // End Cuda or Clang
#else                                            // No cuda support
  #define VoxxHost
  #define VoxxDevice
  #define VoxxDeviceHost
#endif                                           // End VoxxCudaSupported