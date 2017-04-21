//==--- Voxel/SystemInfo/SystemInfo.hpp -------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  SystemInfo.hpp
/// \brief This file defines functionality to get system information.
//
//==------------------------------------------------------------------------==//

#pragma once

namespace Voxx    {
namespace System  {

/// The printAllInfo function prints the information for all components of
/// the system.
/// \todo Add parameter to specific the mode of writing.
void writeSystemInfo();

}} // namespace Voxx::System