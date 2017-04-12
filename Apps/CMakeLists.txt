#==--- Apps/CMakeLists.txt --------------------------------------------------==#

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

include_directories(
  ${Utility_SOURCE_DIR}/Include
  ${VOXX_ROOT}/Include
  ${CUDA_INCLUDE_DIRS}
)

link_directories(${CUDA_LIBRARY_DIRS})

add_executable(
  SystemInformation 
  ${Utility_SOURCE_DIR}/Apps/SystemInformation.cpp
  ${Utility_SOURCE_DIR}/Src/SystemInfo.cpp
)
target_link_libraries(SystemInformation ${CUDA_LIBS})
install(TARGETS SystemInformation DESTINATION ${CMAKE_INSTALL_PREFIX}/Voxel/bin)