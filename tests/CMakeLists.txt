#==--- tests/CMakeLists.txt -------------------------------------------------==#

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS}
                    ${Voxel_SOURCE_DIR}/include)
link_directories(${CUDA_LIBRARY_DIRS})

voxx_add_definitions(${VOXX_DEFINITIONS})
voxx_include_directories(${GTEST_INCLUDE_DIRS}
                         ${CUDA_INCLUDE_DIRS}
                         ${Voxel_SOURCE_DIR}/include)
voxx_library_directories(${CUDA_LIBRARY_DIRS})

voxx_add_executable(AlgorithmTests ${Voxel_SOURCE_DIR}/tests/AlgorithmTests.cpp)
voxx_target_link_libraries(AlgorithmTests ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBS})

voxx_add_executable(ContainerTests ${Voxel_SOURCE_DIR}/tests/ContainerTests.cpp)
voxx_target_link_libraries(ContainerTests ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBS})

voxx_add_executable(FunctionTests
                    ${Voxel_SOURCE_DIR}/tests/FunctionTests.cpp
                    ${Voxel_SOURCE_DIR}/src/Heterogeneous/HeteroException.cpp)
voxx_target_link_libraries(FunctionTests ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBS})

voxx_add_executable(HeterogeneousTests
                    ${Voxel_SOURCE_DIR}/tests/HeterogeneousTests.cu
                    ${Voxel_SOURCE_DIR}/src/Heterogeneous/HeteroException.cpp)
voxx_target_link_libraries(HeterogeneousTests
                           ${GTEST_BOTH_LIBRARIES}
                           ${CUDA_LIBS})

voxx_add_executable(MathTests ${Voxel_SOURCE_DIR}/tests/MathTests.cpp)
voxx_target_link_libraries(MathTests ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBS})

voxx_add_executable(IteratorTests ${Voxel_SOURCE_DIR}/tests/IteratorTests.cpp)
voxx_target_link_libraries(IteratorTests ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBS})

#==--- Benchmarks -----------------------------------------------------------==#

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /opt/google)
find_package(benchmark)
if (benchmark_FOUND)
  set(benchmark_ROOT         ${benchmark_DIR}/../../..)
  set(benchmark_INCLUDE_DIRS ${benchmark_ROOT}/include)
  set(benchmark_LIBRARY_DIRS ${benchmark_ROOT}/lib    )
  set(benchmark_LIBRARIES    -lbenchmark -pthread     )

  voxx_library_directories_append(${benchmark_LIBRARY_DIRS})
  voxx_include_directories_append(${benchmark_INCLUDE_DIRS})

  voxx_add_executable(AlgorithmBenchmarks
                      ${Voxel_SOURCE_DIR}/tests/AlgorithmBenchmarks.cpp)
  voxx_target_link_libraries(AlgorithmBenchmarks
                             ${Voxel_LIBRARIES}
                             ${benchmark_LIBRARIES} -lc++abi)
endif(benchmark_FOUND)