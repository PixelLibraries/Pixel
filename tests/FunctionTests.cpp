//==--- tests/FunctionTests.hpp ---------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file   FunctionTests.cpp
/// \brief  This file defines tests for the function module.
// 
//==------------------------------------------------------------------------==//

#include <Voxel/Function/Callable.hpp>
#include <gtest/gtest.h>

using namespace Voxx::Function;

TEST(CallableTests, CallableCaptureCorrect) {
  int x = 4;
  auto callable = makeCallable([&x] { x *= 2; });

  callable();
  EXPECT_EQ(x, 8);
}

TEST(CallableTests, CallableCanHandleReferences) {
  int x = 4;
  auto callable = makeCallable([] (auto&& val) {
    val *= 2;
  }, x);
  callable();

  EXPECT_EQ(x, 8);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
