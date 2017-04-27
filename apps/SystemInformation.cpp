//==--- Apps/SystemInformation.cpp ------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  SystemInformation.cpp
/// \brief This file defines the implementation of an application which prints
///        the information for the system.
//
//==------------------------------------------------------------------------==//

#include <Voxel/SystemInfo/SystemInfo.hpp>
#include <Voxel/SystemInfo/CpuInfo.hpp>

int main(int argc, char** argv) {
#if defined(__APPLE__)
  Voxx::System::writeSystemInfo();
#else
  auto registers = Voxx::System::CpuInfo::Detail::cpuid(1);
  registers.print();
#endif
  return 0;
}