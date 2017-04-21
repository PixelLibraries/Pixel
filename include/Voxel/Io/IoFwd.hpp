//==--- Voxel/Io/IoFwd.hpp --------------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  IoFwd.hpp
/// \brief This file forward declares Input/Output functionality.
//
//==------------------------------------------------------------------------==//

#pragma once

#include <cstdint>

namespace Voxx    {
namespace Io      {

/// The Format enum defines the types of formats for input and output.
enum class Format : uint8_t {
  None   = 0x00,    //!< No formatting.
  Hex    = 0x01,    //!< Hexadecimal formatting.
  Binary = 0x02     //!< Binary formatting.
};

/// The Mode struct defines modes of input and output.
enum class Mode : uint16_t {
  Console = 0x0000,   //!< Output to the console.
  File    = 0x0001   //!< Output to a file.
};

/// The Output struct provides definitions for different types of output.
/// \tparam  OutputMode  The mode of output.
template <Mode OutputMode>
struct Output;

}} // namespace Voxx::Io