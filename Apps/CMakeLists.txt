#==--- Utility/apps/CMakeLists.txt ------------------------------------------==#

#==--- Cuda -----------------------------------------------------------------==#

find_package(CUDA)
if(CUDA_FOUND)
  if(APPLE)
    set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    set(CUDA_LIBS  -lcudart_static -ldl -pthread      )
  else()
    set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    set(CUDA_LIBS  -lcudart_static -ldl -lrt -pthread   )
  endif()

  # TODO: Get the sm argument generically  
  set(CUDA_FLAGS --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} --cuda-gpu-arch=sm_20)
else()
  message("\nWarning:\n\tCuda not found!\n")
  set(CUDA_FLAGS        )
  set(CUDA_INCLUDE_DIRS )
  set(CUDA_LIBRARY_DIRS )
  set(CUDA_LIBS         )
endif(CUDA_FOUND)

#==--- Common ---------------------------------------------------------------==#

include_directories(${Voxel_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

#==--- Apps -----------------------------------------------------------------==#

add_executable(
  SystemInformation 
  ${Voxel_SOURCE_DIR}/apps/SystemInformation.cpp
  ${Voxel_SOURCE_DIR}/src/SystemInfo.cpp)

target_link_libraries(SystemInformation ${CUDA_LIBS})
install(TARGETS              SystemInformation
        EXPORT               VoxelTargets
        LIBRARY DESTINATION  lib
        ARCHIVE DESTINATION  lib
        RUNTIME DESTINATION  bin
        INCLUDES DESTINATION include)