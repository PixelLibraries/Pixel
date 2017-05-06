//==--- tests/ContainerTests.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//  
//==------------------------------------------------------------------------==//
//
/// \file  ContainerTests.cpp
/// \brief This file defines tests for containers.
// 
//==------------------------------------------------------------------------==//

#include <Voxel/Container/Tuple.hpp>
#include <gtest/gtest.h>
#include <string>

using namespace Voxx;

TEST(TupleTests, CanCreateTupleWithConstructor) {
  Tuple<int, float, std::string> tuple(4, 3.5f, "test");

  EXPECT_EQ(decltype(tuple)::elements, 3     );
  EXPECT_EQ(get<0>(tuple)            , 4     );
  EXPECT_EQ(get<1>(tuple)            , 3.5   );
  EXPECT_EQ(get<2>(tuple)            , "test");
}


TEST(TupleTests, CanCopyConstructTuples) {
  auto       tuple1 = make_tuple(4, 3.5, "some value");
  const auto tuple2 = make_tuple(4, 3.5, "some value");

  EXPECT_EQ(decltype(tuple1)::elements, 3           );
  EXPECT_EQ(get<0>(tuple1)            , 4           );
  EXPECT_EQ(get<1>(tuple1)            , 3.5         );
  EXPECT_EQ(get<2>(tuple1)            , "some value"); 
  EXPECT_EQ(decltype(tuple2)::elements, 3           );
  EXPECT_EQ(get<0>(tuple2)            , 4           );
  EXPECT_EQ(get<1>(tuple2)            , 3.5         );
  EXPECT_EQ(get<2>(tuple2)            , "some value");

  auto       tuple3(tuple1);
  const auto tuple4(tuple2);

  EXPECT_EQ(decltype(tuple1)::elements, 3           );
  EXPECT_EQ(get<0>(tuple1)            , 4           );
  EXPECT_EQ(get<1>(tuple1)            , 3.5         );
  EXPECT_EQ(get<2>(tuple1)            , "some value");
  EXPECT_EQ(decltype(tuple2)::elements, 3           );
  EXPECT_EQ(get<0>(tuple2)            , 4           );
  EXPECT_EQ(get<1>(tuple2)            , 3.5         );
  EXPECT_EQ(get<2>(tuple2)            , "some value");

  EXPECT_EQ(decltype(tuple3)::elements, 3           );
  EXPECT_EQ(get<0>(tuple3)            , 4           );
  EXPECT_EQ(get<1>(tuple3)            , 3.5         );
  EXPECT_EQ(get<2>(tuple3)            , "some value");
  EXPECT_EQ(decltype(tuple4)::elements, 3           );
  EXPECT_EQ(get<0>(tuple4)            , 4           );
  EXPECT_EQ(get<1>(tuple4)            , 3.5         );
  EXPECT_EQ(get<2>(tuple4)            , "some value"); 
}

TEST(TupleTests, CanMoveConstructTuples) {
  auto tuple1 = make_tuple(4, 3.5, "some value");
  auto tuple2 = make_tuple(4, 3.5, "some value");

  EXPECT_EQ(decltype(tuple1)::elements, 3           );
  EXPECT_EQ(get<0>(tuple1)            , 4           );
  EXPECT_EQ(get<1>(tuple1)            , 3.5         );
  EXPECT_EQ(get<2>(tuple1)            , "some value"); 
  EXPECT_EQ(decltype(tuple2)::elements, 3           );
  EXPECT_EQ(get<0>(tuple2)            , 4           );
  EXPECT_EQ(get<1>(tuple2)            , 3.5         );
  EXPECT_EQ(get<2>(tuple2)            , "some value");

  auto       tuple3(std::move(tuple1));
  const auto tuple4(std::move(tuple2));

  EXPECT_EQ(decltype(tuple1)::elements, 3           );
  EXPECT_EQ(get<0>(tuple1)            , 4           );
  EXPECT_EQ(get<1>(tuple1)            , 3.5         );
  EXPECT_EQ(get<2>(tuple1)            , "some value");
  EXPECT_EQ(decltype(tuple2)::elements, 3           );
  EXPECT_EQ(get<0>(tuple2)            , 4           );
  EXPECT_EQ(get<1>(tuple2)            , 3.5         );
  EXPECT_EQ(get<2>(tuple2)            , "some value");

  EXPECT_EQ(decltype(tuple3)::elements, 3           );
  EXPECT_EQ(get<0>(tuple3)            , 4           );
  EXPECT_EQ(get<1>(tuple3)            , 3.5         );
  EXPECT_EQ(get<2>(tuple3)            , "some value");
  EXPECT_EQ(decltype(tuple4)::elements, 3           );
  EXPECT_EQ(get<0>(tuple4)            , 4           );
  EXPECT_EQ(get<1>(tuple4)            , 3.5         );
  EXPECT_EQ(get<2>(tuple4)            , "some value"); 
}

TEST(TupleTests, CanCopyAssignTuples) {
  auto       tuple1 = make_tuple(4, 3.5, "some value");
  const auto tuple2 = make_tuple(4, 3.5, "some value");

  EXPECT_EQ(decltype(tuple1)::elements, 3           );
  EXPECT_EQ(get<0>(tuple1)            , 4           );
  EXPECT_EQ(get<1>(tuple1)            , 3.5         );
  EXPECT_EQ(get<2>(tuple1)            , "some value"); 
  EXPECT_EQ(decltype(tuple2)::elements, 3           );
  EXPECT_EQ(get<0>(tuple2)            , 4           );
  EXPECT_EQ(get<1>(tuple2)            , 3.5         );
  EXPECT_EQ(get<2>(tuple2)            , "some value");

  auto       tuple3 = tuple1;
  const auto tuple4 = tuple2;

  EXPECT_EQ(decltype(tuple1)::elements, 3           );
  EXPECT_EQ(get<0>(tuple1)            , 4           );
  EXPECT_EQ(get<1>(tuple1)            , 3.5         );
  EXPECT_EQ(get<2>(tuple1)            , "some value");
  EXPECT_EQ(decltype(tuple2)::elements, 3           );
  EXPECT_EQ(get<0>(tuple2)            , 4           );
  EXPECT_EQ(get<1>(tuple2)            , 3.5         );
  EXPECT_EQ(get<2>(tuple2)            , "some value");

  EXPECT_EQ(decltype(tuple3)::elements, 3           );
  EXPECT_EQ(get<0>(tuple3)            , 4           );
  EXPECT_EQ(get<1>(tuple3)            , 3.5         );
  EXPECT_EQ(get<2>(tuple3)            , "some value");
  EXPECT_EQ(decltype(tuple4)::elements, 3           );
  EXPECT_EQ(get<0>(tuple4)            , 4           );
  EXPECT_EQ(get<1>(tuple4)            , 3.5         );
  EXPECT_EQ(get<2>(tuple4)            , "some value"); 
}

TEST(TupleTests, CanMoveAssignTuples) {
  auto tuple1 = make_tuple(4, 3.5, "some value");
  auto tuple2 = make_tuple(4, 3.5, "some value");

  EXPECT_EQ(decltype(tuple1)::elements, 3           );
  EXPECT_EQ(get<0>(tuple1)            , 4           );
  EXPECT_EQ(get<1>(tuple1)            , 3.5         );
  EXPECT_EQ(get<2>(tuple1)            , "some value"); 
  EXPECT_EQ(decltype(tuple2)::elements, 3           );
  EXPECT_EQ(get<0>(tuple2)            , 4           );
  EXPECT_EQ(get<1>(tuple2)            , 3.5         );
  EXPECT_EQ(get<2>(tuple2)            , "some value");

  auto       tuple3 = std::move(tuple1);
  const auto tuple4 = std::move(tuple2);

  EXPECT_EQ(decltype(tuple1)::elements, 3           );
  EXPECT_EQ(get<0>(tuple1)            , 4           );
  EXPECT_EQ(get<1>(tuple1)            , 3.5         );
  EXPECT_EQ(get<2>(tuple1)            , "some value");
  EXPECT_EQ(decltype(tuple2)::elements, 3           );
  EXPECT_EQ(get<0>(tuple2)            , 4           );
  EXPECT_EQ(get<1>(tuple2)            , 3.5         );
  EXPECT_EQ(get<2>(tuple2)            , "some value");

  EXPECT_EQ(decltype(tuple3)::elements, 3           );
  EXPECT_EQ(get<0>(tuple3)            , 4           );
  EXPECT_EQ(get<1>(tuple3)            , 3.5         );
  EXPECT_EQ(get<2>(tuple3)            , "some value");
  EXPECT_EQ(decltype(tuple4)::elements, 3           );
  EXPECT_EQ(get<0>(tuple4)            , 4           );
  EXPECT_EQ(get<1>(tuple4)            , 3.5         );
  EXPECT_EQ(get<2>(tuple4)            , "some value"); 
}

TEST(TupleTests, CanDeduceTupleTypeWithMakeTuple) {
  auto tuple = make_tuple(4, 3.5, "some value");

  EXPECT_EQ(decltype(tuple)::elements, 3           );
  EXPECT_EQ(get<0>(tuple)            , 4           );
  EXPECT_EQ(get<1>(tuple)            , 3.5         );
  EXPECT_EQ(get<2>(tuple)            , "some value");
}

TEST(TupleTests, AtWrapperCanAccessElements) {
  auto tuple = make_tuple(4, 3.5, "some value");

  EXPECT_EQ(tuple.at<0>(), 4           );
  EXPECT_EQ(tuple.at<1>(), 3.5         );
  EXPECT_EQ(tuple.at<2>(), "some value");
}

TEST(TupleTests, AtWrapperCanSetElements) {
  volatile auto tuple = make_tuple(4, 3.5, "some value");

  EXPECT_EQ(tuple.at<0>(), 4           );
  EXPECT_EQ(tuple.at<1>(), 3.5         );
  EXPECT_EQ(tuple.at<2>(), "some value");

  tuple.at<0>() = 7;
  tuple.at<1>() = 14.4;
  tuple.at<2>() = "another value";

  EXPECT_EQ(tuple.at<0>(), 7              );
  EXPECT_EQ(tuple.at<1>(), 14.4           );
  EXPECT_EQ(tuple.at<2>(), "another value");
}

template <std::size_t I, typename TupleType>
using TupElemT = typename tuple_element<I, TupleType>::type;

TEST(TupleTests, TupleElementWorks) {
  Tuple<int, float, double, std::string> tuple(4, 3.5f, 4.0, "test");

  bool same1 = std::is_same<TupElemT<0, decltype(tuple)>, int&&>::value;
  bool same2 = std::is_same<TupElemT<1, decltype(tuple)>, float&&>::value;
  bool same3 = std::is_same<TupElemT<2, decltype(tuple)>, double&&>::value;
  bool same4 = std::is_same<TupElemT<3, decltype(tuple)>, std::string&&>::value;

  EXPECT_TRUE(same1);
  EXPECT_TRUE(same2);
  EXPECT_TRUE(same3);
  EXPECT_TRUE(same4);
}

TEST(TupleTests, CanCreateReferenceContainer) {
  int x = 0; float y = 0.0f;
  Tuple<int&, float&> tuple(x, y);

  EXPECT_EQ(x            , 0   );
  EXPECT_EQ(y            , 0.0f);
  EXPECT_EQ(tuple.at<0>(), 0   );
  EXPECT_EQ(tuple.at<1>(), 0.0f);

  get<0>(tuple) = 4;
  tuple.at<1>() = 3.5f;

  EXPECT_EQ(x            , 4   );
  EXPECT_EQ(y            , 3.5f);
  EXPECT_EQ(tuple.at<0>(), 4   );
  EXPECT_EQ(tuple.at<1>(), 3.5f);

  x = 0; y = 0.0f;
  EXPECT_EQ(x            , 0   );
  EXPECT_EQ(y            , 0.0f);
  EXPECT_EQ(tuple.at<0>(), 0   );
  EXPECT_EQ(tuple.at<1>(), 0.0f);

}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}