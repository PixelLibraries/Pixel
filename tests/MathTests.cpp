//==--- tests/MathTests.hpp -------------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  MathTests.cpp
/// \brief This file defines tests for the math module.
// 
//==------------------------------------------------------------------------==//

#include <Voxel/Math/Math.hpp>
#include <gtest/gtest.h>

using namespace Voxx::Math;

TEST(MathTests, PowerOfTwoTestCorrect) {
  EXPECT_FALSE(isPowerOfTwo(0));
  EXPECT_TRUE(isPowerOfTwo(1));

  for (std::size_t i = 1; i < sizeof(std::size_t) * 8; ++i) {
    auto first  = isPowerOfTwo(1ull << i);
    auto second = isPowerOfTwo((1ull << i) + 1);

    EXPECT_TRUE(first);
    EXPECT_FALSE(second);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}