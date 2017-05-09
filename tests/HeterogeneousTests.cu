//==--- tests/HeterogeneousTests.cu ------------------------ -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  HeterogeneousTests.cu
/// \brief This file defines tests for the heterogeneous components.
// 
//==------------------------------------------------------------------------==//

#include <Voxel/Heterogeneous/Launch.hpp>
#include <gtest/gtest.h>

using namespace Voxx::Hetero;

TEST(HeterogeneousTests, CanLaunchLambdaOnDeviceWithSimpleTypes) {
  int x = 7;
  EXPECT_EQ(x, 7);

  launch(dim3(1), dim3(1), [] VoxxDeviceHost (auto&& value) {
    value = 1;
  }, x);

  EXPECT_EQ(x, 1);
}

TEST(HeterogeneousTests, CanLaunchLambdaOnDeviceWithDifferentTypes) {
  int x = 7; const float y = 4.0f;
  EXPECT_EQ(x, 7   );
  EXPECT_EQ(y, 4.0f);

  launch(dim3(1), dim3(1), [] VoxxDeviceHost (auto&& a, const auto& b) {
    a = 1 + static_cast<std::decay_t<decltype(a)>>(b);
  }, x, y);

  EXPECT_EQ(x, 5   );
  EXPECT_EQ(y, 4.0f);
}

// This test doesn't check anything, just tests that the rvalue's passed to
// launch don't cause a compiler or runtime error.
TEST(HeterogeneousTests, CanLaunchLambdaOnDeviceWithOnlyRvalues) {
  launch(dim3(1), dim3(1), [] VoxxDeviceHost (auto&& a, auto&& b) {
    a *= 2; b *= b;
  }, 10, 14.0f);
}

TEST(HeterogeneousTests, CanLaunchLambdaOnDeviceWithRvalues) {
  int x = 7;
  EXPECT_EQ(x, 7   );

  launch(dim3(1), dim3(1), [] VoxxDeviceHost (auto&& a, auto&& b) {
    a = b;
  }, x, 12);

  EXPECT_EQ(x, 12);
}

// NOTE: We cannot capture by reference, cuda does not allow that:
TEST(HeterogeneousTests, CanLaunchLambdaWithCaptures) {
  int a = 0;
  int x = 4;
  launch(dim3(1), dim3(1), [x] VoxxDeviceHost (auto&& var) {
    var = x;
  }, a);
  EXPECT_EQ(a, 4);
  EXPECT_EQ(x, 4);

  // Don't do this, it will cause a memory access error:
/*
  launch(dim3(1), dim3(1), [&x] VoxxDeviceHost (auto&& var) {
    var = x;
  }, a);
  EXPECT_EQ(a, 4); 
*/
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}