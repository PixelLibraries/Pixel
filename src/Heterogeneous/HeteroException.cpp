//==--- stc/Heterogeneous/HeteroException.cpp -------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  HeteroException.cpp
/// \brief This file provides an implementation for heterogenous exception
///        functionality
//
//==------------------------------------------------------------------------==//

#include <Voxel/Heterogeneous/HeteroException.hpp>

namespace Voxx   {
namespace Hetero {

const char* HeteroException::message() const noexcept {
  switch (ErrorType) {
    case Type::Realloc     : return "Reallocating non null pointer!";
    case Type::InvalidCopy : return "Invalid copy between device and host";
    default                : return "Unknown heterogeneous error";
  }
}

}} // namespace Vox::Hetero
