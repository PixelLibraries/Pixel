#==--- Utility/src/CMakeLists.txt -------------------------------------------==#

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

include_directories(${CUDA_INCLUDE_DIRS} ${Voxel_SOURCE_DIR}/include)

#==--- Libraries ------------------------------------------------------------==#

add_library(VoxelSystemInfo SystemInfo.cpp)
install(TARGETS  VoxelSystemInfo
        EXPORT   VoxelLibs
        LIBRARY  DESTINATION lib
        RUNTIME  DESTINATION bin)
target_include_directories(
  VoxelSystemInfo PUBLIC ${CMAKE_INSTALL_PREFIX}/include)
