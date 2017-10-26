//==--- tests/AlgorithmTests.cpp --------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  AlgorithmTests.cpp
/// \brief This file defines tests for the algorithmic functionality.
// 
//==------------------------------------------------------------------------==//

#include <Voxel/Algorithm/Algorithm.hpp>
#include <gtest/gtest.h>
#include <string>

using namespace Voxx;

TEST(ForEachTests, CanUseForEachOnTuples) {
  Tuple<int, float, std::string> tuple(4, 3.5f, "test");

  forEach(tuple, [] (auto& value) {
    value += value;
  });

  EXPECT_EQ(get<0>(tuple), 4 + 4                                    );
  EXPECT_EQ(get<1>(tuple), 3.5f + 3.5f                              );
  EXPECT_EQ(get<2>(tuple), std::string("test") + std::string("test"));
}

TEST(ForEachTests, CanUseForEachOnContainers) {
  std::vector<int> v(100);

  int x = 0;
  forEach(v, [&x] (auto& value) {
    value = x++;
  });

  std::size_t i = 0;
  for (const auto& e : v) {
    EXPECT_EQ(e, i++);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}