//==--- Voxel/Thread/Detail/ThreadImpl.hpp ----------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  ThreadImpl.hpp
/// \brief This file defines thread related implementations.
// 
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Utility/Portability.hpp>

#if defined(VoxxCompilerMSVC)                           // MSVC
  #if defined(VoxxSystemTargetXbox360)                  // XBox
    // According to Microsoft's lockless programming article, _lwsync() acts as
    // a memory and compiler barrier, see 
    // 
    //  https://msdn.microsoft.com/en-us/library/windows/desktop/ee418650.aspx
    //  
    // and specifically the sections on preventing compiler and memory
    // reordering.
    // X360 has weak ordering, so the memory barrier needs to be a full hardware
    // barrier.
    #define VoxxMemoryBarrier() __lwsync()
  #elif defined(VoxxSystemTargetWindows)              // Windows
    // Here a full memory barrier is not needed
    // on x86, since it has stong memory ordering.
    #define VoxxMemoryBarrier() _ReadWriteBarrier()
  #endif                                              // End Xbox360 / Windows
#elif defined(VoxxCompilerClang) || defined(VoxxCompilerGcc)  // Clang | GCC
  #if defined(VoxxCpuArchArm)                                 // ARM
    #if VoxxCpuArchArm >= 7                                   // ARM V7
      // For armV7 and above, the dmb ish instruction creates a memory barrier,
      // and since it is a weakly ordered archtitecture, a full barrier is
      // required.
      #define VoxxMemoryBarrier() asm volatile("dmb ish" ::: "memory")
    #elif VoxxCpuArchArm == 6                                 // ARM V6
      #define VoxxMemoryBarrier()                                              \
        asm volatile("mcr p15, 0, %0, c7, c10, 5" :: "r"(0) : "memory")
    #else
      // Just use a non-inlined function call which acts as a barrier.
      #define VoxxMemoryBarrier() void fakeMemoryBarrier() noexcept;
    #endif                                                    // End Arm version
  #else                                                       // Any other CPU
    // For x86, again only a memory clobber is required.
    #define VoxxMemoryBarrier() asm volatile("" ::: "memory")
  #endif                                                      // End cpu arch 
#endif                                                        // End compiler 