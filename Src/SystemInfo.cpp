//==--- Utility/SystemInfo.cpp ----------------------------- -*- C++ -*- ---==//
//            
//                                Voxel : Utility 
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  SystemInfo.cpp
/// \brief This file get system information and prints the information to the
//         screen.
//
//==------------------------------------------------------------------------==//

#include <Voxel/Utility/SystemInfo/SystemInfo.hpp>

int main(void) {
  auto systemInfo = Voxx::Utility::SystemInfo();
  systemInfo.print();
  return 0;
}