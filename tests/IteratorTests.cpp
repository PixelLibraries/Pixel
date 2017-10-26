//==--- tests/IteratorTests.cpp ---------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  IteratorTests.cpp
/// \brief This file defines tests for iterator functionality.
// 
//==------------------------------------------------------------------------==//

#include <Voxel/Iterator/Iterators.hpp>
#include <gtest/gtest.h>
#include <string>

using namespace Voxx;

TEST(RangeTests, CanCreateDefaultIntegerRange) {
  int value = 0;
  for (auto i : range(0, 100)) {
    EXPECT_EQ(i, value++);
  }
  value = 0;
  for (auto i : range(0, 200, 3)) {
    EXPECT_EQ(i, value);
    value += 3;
  }
}

TEST(RangeTests, CanCreateFloatingPointRange) {
  float value = 0.f;
  for (auto i : range(0.f, 10.f)) {
    EXPECT_EQ(i, value);
    value += 1.0f;
  }
  value = 0.0f;
  for (auto i : range(0.f, 1.5f, .1f)) {
    EXPECT_EQ(i, value);
    value += .1f;
  }
}

TEST(RangeTests, CanCreateDoublePrecisionRange) {
  double value = 0.0;
  for (auto i : range(0.0, 10.0)) {
    EXPECT_EQ(i, value);
    value += 1.0;
  }
  value = 0.0;
  for (auto i : range(0.0, 2.5, 0.2)) {
    EXPECT_EQ(i, value);
    value += 0.2;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}