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