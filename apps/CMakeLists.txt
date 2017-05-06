#==--- Utility/apps/CMakeLists.txt ------------------------------------------==#

include_directories(${Voxel_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

#==--- Apps -----------------------------------------------------------------==#

add_executable(
  SystemInformation 
  ${Voxel_SOURCE_DIR}/apps/SystemInformation.cpp
  ${Voxel_SOURCE_DIR}/src/System/SystemInfo.cpp
  ${Voxel_SOURCE_DIR}/src/Thread/Thread.cpp)

target_link_libraries(SystemInformation ${CUDA_LIBS})
install(TARGETS              SystemInformation
        EXPORT               VoxelTargets
        ARCHIVE DESTINATION  lib
        LIBRARY DESTINATION  lib
        RUNTIME DESTINATION  bin
        INCLUDES DESTINATION include)