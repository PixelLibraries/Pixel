//==--- Apps/SystemInformation.cpp ------------------------- -*- C++ -*- ---==//
//            
//                                Voxel : Utility 
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

#include <Voxel/Utility/SystemInfo/SystemInfo.hpp>

int main(int argc, char** argv) {
  Voxx::Utility::System::writeSystemInfo();
  return 0;
}