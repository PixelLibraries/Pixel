//==--- Voxel/Heterogeneous/HetroException.hpp ---------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  HeteroException.hpp
/// \brief This file defines a class for heterogeneous exceptions.
//
//==------------------------------------------------------------------------==//

#pragma once

#include <exception>
#include <string>

namespace Voxx   {
/// This namespace containts heterogeneous functionality.
namespace Hetero {

/// The HeteroException class overrides the standard exception class to define
/// heterogeneous related exceptions.
class HeteroException : public std::exception {
 public:
  /// The Type enum defines the types of heterogeneous related exceptions.
  enum class Type : uint8_t {
    Realloc     = 0x00, //!< Reallocation of non null pointer.
    InvalidCopy = 0x01
  };

  /// Constructor -- initializes the type of the heterogeneous error.
  /// \param[in]  type The type of the heterogeneous error.
  HeteroException(Type type) noexcept : ErrorType(type) {}

  /// Displays a message for the exception.
  virtual const char* what() const noexcept override {
    const std::string msg = std::string("\nHeterogeneous Exception:\n\t")
                          + message();
    return msg.c_str();
  }

 private:
  Type ErrorType;    //!< The type of the thread exception.
                  
  /// Returns char string pointer with an appropriate message for the type of
  /// the error.
  const char* message() const noexcept;
};

}} // namespace Voxx::Hetero
