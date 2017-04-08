#==--- Src/CMakeLists.txt ---------------------------------------------------==#

#==--- Cuda -----------------------------------------------------------------==#

FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
  IF(APPLE)
    SET(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    SET(CUDA_LIBS  -lcudart_static -ldl -pthread      )
  ELSE()
    SET(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    SET(CUDA_LIBS  -lcudart_static -ldl -lrt -pthread   )
  ENDIF()

  # TODO: Get the sm argument generically  
  SET(CUDA_FLAGS --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} --cuda-gpu-arch=sm_20)
ELSE()
  SET(CUDA_FLAGS        )
  SET(CUDA_INCLUDE_DIRS )
  SET(CUDA_LIBRARY_DIRS )
  SET(CUDA_LIBS         )
ENDIF(CUDA_FOUND)

#==--- Common ---------------------------------------------------------------==#

INCLUDE_DIRECTORIES(
  ${Utility_SOURCE_DIR}/Include
  ${VOXEL_ROOT}/Include
  ${CUDA_INCLUDE_DIRS}
)

LINK_DIRECTORIES(${CUDA_LIBRARY_DIRS})

ADD_EXECUTABLE(SystemInfo ${Utility_SOURCE_DIR}/Src/SystemInfo.cpp)
TARGET_LINK_LIBRARIES(SystemInfo ${CUDA_LIBS})

INSTALL(TARGETS SystemInfo DESTINATION ${VOXEL_ROOT}/bin)