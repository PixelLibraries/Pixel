//==--- Voxel/Io/Io.hpp ------------------------------------ -*- C++ -*- ---==//
//            
//                                     Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  Io.hpp
/// \brief This file forward implements Input/Output functionality.
//
//==------------------------------------------------------------------------==//

#pragma once

#include "IoFwd.hpp"
#include <iostream>
#include <iomanip>
#include <string>

namespace Voxx    {
/// The Io namespace defines input and output related functionality.
namespace Io      {

/// Specialization for Console output.
template <>
struct Output<Mode::Console> {
  //==--- Constants --------------------------------------------------------==//
  
  /// Defines the width of the terminal.
  static constexpr auto width     = 80;
  /// Defines the width when printing names.
  static constexpr auto nameWidth = 25;
  /// Defines the decorator.
  static constexpr auto decorator = "|";

  /// Defines a function to print a banner.
  static void banner(std::string name = "") {
    // 9 from: 2 * banner; 2 * "=="; 1 * "---"
    auto fillSize = width - 9;
    if (name.size()) {
      // 2 for the spaces on either size of the name.
      fillSize -= name.size() + 2;
      name      = std::string(" ") + name + std::string(" ");
    }
    std::cout << decorator << "==---"
              << name
              << std::string(fillSize, '-')
              << "==" << decorator << "\n";
  }

  //==--- Methods ----------------------------------------------------------==//
  
  /// Defines a function to print a result and the name of the result.
  /// \param[in]  name         The name of the result.
  /// \param[in]  value        The value of the result.
  /// \tparam     Name         The type of the name.
  /// \tparam     Result       The type :::;of the result.
  /// \tparam     ResultFormat The formatting for the result.
  template <typename Name, typename Result, Format ResultFormat = Format::None>
  static void writeResult(Name&& name, Result&& result) {
    constexpr auto resultSize = width - 25 - 9;
    std::cout << std::left             << decorator << " " 
              << std::setw(nameWidth)  << std::left << name
              << std::setw(5)          << std::left << " : " 
              << std::setw(resultSize) << std::right;
    format(std::forward<Result>(result), FormatTag<ResultFormat>{});
    std::cout << " " << decorator << "\n";
  }

 private:
  /// The FormatTag struct defines a struct which can be used to select the
  /// appropriate overload for formatting functions.
  template <Format>
  struct FormatTag {};

  /// Formats the \p value using ValueFormat formatting. This overload is
  /// selected for all formats other than hex.
  /// \param[in]  value         The value to format.
  /// \param[in]  tag           The tag used to select this overload.
  /// \tparam     Value         The type of the \p value.
  /// \tparam     ValueFormat   The type of formatting.
  template <typename Value, Format ValueFormat>
  static void format(Value&& value, FormatTag<ValueFormat> tag) {
      std::cout << value;
  }

  /// Formats the \p value using hex formatting. This overload is
  /// selected for Format::Hex formatting.
  /// \param[in]  value         The value to format.
  /// \param[in]  tag           The tag used to select this overload.
  /// \tparam     Value         The type of the \p value.
  template <typename Value>
  static void format(Value&& value, FormatTag<Format::Hex> tag) {
    std::cout << std::hex << value << std::dec;      
  }
};

}} // namespace Voxx::Io