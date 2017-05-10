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

TEST(TupleTests, CanAccessElements) {
  auto tuple = make_tuple(4, 3.5, "some value");

  EXPECT_EQ(get<0>(tuple), 4           );
  EXPECT_EQ(get<1>(tuple), 3.5         );
  EXPECT_EQ(get<2>(tuple), "some value");
}

TEST(TupleTests, CanSetElements) {
  volatile auto tuple = make_tuple(4, 3.5);

  EXPECT_EQ(get<0>(tuple), 4  );
  EXPECT_EQ(get<1>(tuple), 3.5 );

  get<0>(tuple) = 7;
  get<1>(tuple) = 14.4;

  EXPECT_EQ(get<0>(tuple), 7   );
  EXPECT_EQ(get<1>(tuple), 14.4);
}

TEST(TupleTests, TupleElementWorks) {
  Tuple<int, float, double, std::string> tuple(4, 3.5f, 4.0, "test");

  using Tup = decltype(tuple);
  bool res1 = std::is_same<tuple_element_t<0, Tup>, int>::value;
  bool res2 = std::is_same<tuple_element_t<1, Tup>, float>::value;
  bool res3 = std::is_same<tuple_element_t<2, Tup>, double>::value;
  bool res4 = std::is_same<tuple_element_t<3, Tup>, std::string>::value;

  EXPECT_TRUE(res1);
  EXPECT_TRUE(res2);
  EXPECT_TRUE(res3);
  EXPECT_TRUE(res4);
}

TEST(TupleTests, TupleGetHasCorrectTypes) {
  Tuple<int, float, double, std::string>       tuple1(4, 3.5f, 4.0, "test");
  const Tuple<int, float, double, std::string> tuple2(4, 3.5f, 4.0, "test");

  bool res1 = std::is_same<decltype(get<0>(tuple1)), int&>::value;
  bool res2 = std::is_same<decltype(get<1>(std::move(tuple1))), float&&>::value;
  bool res3 = std::is_same<decltype(get<2>(tuple2)), const double&>::value;

  EXPECT_TRUE(res1);
  EXPECT_TRUE(res2);
  EXPECT_TRUE(res3);
}


TEST(TupleTests, CanCreateReferenceContainer) {
  int x = 0; float y = 0.0f;
  Tuple<int&, float&> tuple(x, y);

  EXPECT_EQ(x            , 0   );
  EXPECT_EQ(y            , 0.0f);
  EXPECT_EQ(get<0>(tuple), 0   );
  EXPECT_EQ(get<1>(tuple), 0.0f);

  get<0>(tuple) = 4;
  get<1>(tuple) = 3.5f;

  EXPECT_EQ(x            , 4   );
  EXPECT_EQ(y            , 3.5f);
  EXPECT_EQ(get<0>(tuple), 4   );
  EXPECT_EQ(get<1>(tuple), 3.5f);

  x = 0; y = 0.0f;
  EXPECT_EQ(x            , 0   );
  EXPECT_EQ(y            , 0.0f);
  EXPECT_EQ(get<0>(tuple), 0   );
  EXPECT_EQ(get<1>(tuple), 0.0f);

}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}