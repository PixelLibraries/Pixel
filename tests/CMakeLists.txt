#==--- tests/CMakeLists.txt -------------------------------------------------==#

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS}
                    ${Voxel_SOURCE_DIR}/include)
link_directories(${CUDA_LIBRARY_DIRS})

add_executable(ContainerTests ContainerTests.cpp)
target_link_libraries(ContainerTests ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBS})