#==--- Utility/CMakeLists.txt -----------------------------------------------==#

CMAKE_MINIMUM_REQUIRED(VERSION 3.4)
PROJECT(Utility)

#==--- Versioning -----------------------------------------------------------==#

SET(Utility_MAJOR_VERSION 0)
SET(Utility_MINOR_VERSION 1)
SET(Utility_PATCH_VERSION 0)
SET(Utility_VERSION
  ${Utility_MAJOR_VERSION}.${Utility_MINOR_VERSION}.${Utility_PATCH_VERSION}
)

#==--- Build Type -----------------------------------------------------------==#

IF(NOT CMAKE_BUILD_TYPE) 
  SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type not specified!" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

#==--- Options --------------------------------------------------------------==#

OPTION(VOXEL_ROOT "Path to Voxel root directory")

SET(CMAKE_CXX_COMPILER clang++-5)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Utility_SOURCE_DIR}/CMake")
INCLUDE(VoxxSystemInfo)

#==--- Packages -------------------------------------------------------------==#

FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
  ADD_DEFINITIONS(-DVoxxCudaSupported)
  SET(CXX_CUDA_FLAGS "-Xcuda-ptxas -v")
ENDIF(CUDA_FOUND)

#==--- Compiler Flags -------------------------------------------------------==#

IF(WIN32)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
  SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} /O2"  )
ELSE()
  SET(FLAGS        "-std=c++1z -O3"                                     )
  SET(DEBUG_FLAGS  "${FLAGS} -Wall -Wextra -Wpedantic ${CXX_CUDA_FLAGS}")
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${FLAGS}"     )
  SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}" )
ENDIF()

#==--- Set CXXFLAGS ---------------------------------------------------------==#

IF(CMAKE_BUILD_TYPE MATCHES Release)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}"
      CACHE STRING "CXX FLAGS" FORCE)
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}"
      CACHE STRING "CXX FLAGS" FORCE)
ENDIF()

#==--- Install --------------------------------------------------------------==#

IF(${VOXEL_ROOT} MATCHES OFF)
  MESSAGE("\nWARNING:\n\t VOXEL_ROOT not specified, using the default.\n")
  IF(WIN32)
    SET(VOXEL_ROOT)
  ELSE()
    SET(VOXEL_ROOT /opt/Voxel)
  ENDIF()
ENDIF()

IF(${CMAKE_INSTALL_PREFIX} MATCHES ${VOXEL_ROOT})
ELSE()
  MESSAGE("\nWARNING:\n\tInstallation prefix does not match VOXEL_ROOT!")
  MESSAGE("\tUse -DCMAKE_INSTALL_PREFIX=/path/to/voxel/root\n")
ENDIF()

INSTALL(DIRECTORY ${Utility_SOURCE_DIR}/Include/Voxel/Utility
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Voxel)

#==--- Subdirectories -------------------------------------------------------==#

ADD_SUBDIRECTORY(Cmake)
ADD_SUBDIRECTORY(Docs)
ADD_SUBDIRECTORY(Src)

#VOXX_SYSTEM_INFO()

#==--- Print Configuration --------------------------------------------------==#

MESSAGE("|==--- CONFIGURATION --------------------------------------------==|")
MESSAGE("| UTILITY VERSION         : ${Utility_VERSION}"                      )
MESSAGE("| BUILD TYPE              : ${CMAKE_BUILD_TYPE}"                     )
MESSAGE("| CMAKE_CXX_FLAGS_RELEASE : ${CMAKE_CXX_FLAGS_RELEASE}"              )
MESSAGE("| CMAKE_CXX_FLAGS_DEBUG   : ${CMAKE_CXX_FLAGS_DEBUG}"                )
MESSAGE("| CMAKE_CXX_FLAGS         : ${CMAKE_CXX_FLAGS}"                      )
MESSAGE("| CMAKE_INSTALL_PREFIX    : ${CMAKE_INSTALL_PREFIX}"                 )
MESSAGE("| VOXEL_ROOT              : ${VOXEL_ROOT}"                           )
MESSAGE("| CUDA SUPPORTED          : ${CUDA_FOUND}"                           )
MESSAGE("| CUDA ROOT               : ${CUDA_TOOLKIT_ROOT_DIR}"                )
MESSAGE("| PHYSICAL CORES          : ${VOXX_PHYSICAL_CORES}"                  )
MESSAGE("| LOGICAL CORES           : ${VOXX_LOGICAL_CORES}"                   )
MESSAGE("| CACHELINE SIZE          : ${VOXX_CACHELINE_SIZE}"                  )
MESSAGE("|==--------------------------------------------------------------==|")
