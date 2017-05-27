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

using namespace Voxx;

TEST(MathTests, PowerOfTwoTestCorrect) {
  EXPECT_FALSE(Math::isPowerOfTwo(0));
  EXPECT_TRUE(Math::isPowerOfTwo(1));

  for (std::size_t i = 1; i < sizeof(std::size_t) * 8; ++i) {
    auto first  = Math::isPowerOfTwo(1ull << i);
    auto second = Math::isPowerOfTwo((1ull << i) + 1);

    EXPECT_TRUE(first);
    EXPECT_FALSE(second);
  }
}

TEST(MathTests, RandIntTestCorrect) {
  auto random1 = Math::randint(0 , 100);
  auto random2 = Math::randint(10, 1000);

  EXPECT_GE(random1, 0   );
  EXPECT_LE(random1, 100 );
  EXPECT_GE(random2, 10  );
  EXPECT_LE(random2, 1000);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}