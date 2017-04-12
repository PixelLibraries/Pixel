#==--- Utility/CMakeLists.txt -----------------------------------------------==#

cmake_minimum_required(VERSION 3.4)
project(Utility)

#==--- Versioning -----------------------------------------------------------==#

set(Utility_MAJOR_VERSION 0)
set(Utility_MINOR_VERSION 1)
set(Utility_PATCH_VERSION 0)
set(Utility_VERSION
  ${Utility_MAJOR_VERSION}.${Utility_MINOR_VERSION}.${Utility_PATCH_VERSION}
)

#==--- Build Type -----------------------------------------------------------==#

if(NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type not specendified!" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

#==--- Options --------------------------------------------------------------==#

option(VOXX_ROOT "Path to Voxel root directory")

set(CMAKE_CXX_COMPILER clang++-5)
set(CMAKE_MODULE_PATH 
      ${CMAKE_MODULE_PATH}
      ${Utility_SOURCE_DIR}/Cmake
)
include(VoxxSystemInfo)

#==--- Packages -------------------------------------------------------------==#

find_package(CUDA)
if(CUDA_FOUND)
  add_definitions(-DVoxxCudaSupported)
  set(CXX_CUDA_FLAGS "-Xcuda-ptxas -v")
endif(CUDA_FOUND)

#==--- Compiler Flags -------------------------------------------------------==#

if(WIN32)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} /O2"  )
else()
  set(FLAGS        "-std=c++1z -O3"                                     )
  set(DEBUG_FLAGS  "${FLAGS} -Wall -Wextra -Wpedantic ${CXX_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${FLAGS}"     )
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}" )
endif()

#==--- set CXXFLAGS ---------------------------------------------------------==#

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}"
      CACHE STRING "CXX FLAGS" FORCE)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "CXX FLAGS" FORCE)
endif()

#==--- Voxel Root -----------------------------------------------------------==#

if(${VOXX_ROOT} MATCHES OFF)
  message("\nWarning:\n\t VOXX_ROOT not specified, using the default.\n")
  if(WIN32)
    set(VOXX_ROOT)
  else()
    set(VOXX_ROOT /opt/Voxel)
  endif()
endif()

#==--- Subdirectories -------------------------------------------------------==#

voxx_system_info()

add_subdirectory(Apps)
add_subdirectory(Cmake)
add_subdirectory(Docs)
add_subdirectory(Src)

#==--- Install --------------------------------------------------------------==#

install(DIRECTORY   ${Utility_SOURCE_DIR}/Include/Voxel/Utility
        DESTINATION ${CMAKE_INSTALL_PREFIX}/Voxel/include/Voxel)

#==--- Print Configuration --------------------------------------------------==#
message("\n|==---------------------==|")
message("| Configuration:          |"  )
message("|==---------------------==|"  )
message("| UTILITY VERSION         : ${Utility_VERSION}"                       )
message("| BUILD TYPE              : ${CMAKE_BUILD_TYPE}"                      )
message("| CMAKE_CXX_FLAGS_RELEASE : ${CMAKE_CXX_FLAGS_RELEASE}"               )
message("| CMAKE_CXX_FLAGS_DEBUG   : ${CMAKE_CXX_FLAGS_DEBUG}"                 )
message("| CMAKE_CXX_FLAGS         : ${CMAKE_CXX_FLAGS}"                       )
message("| INSTALL_PREFIX          : ${CMAKE_INSTALL_PREFIX}/Voxel"            )
message("| VOXEL_ROOT              : ${VOXX_ROOT}"                             )
message("| CUDA SUPPORTED          : ${CUDA_FOUND}"                            )
message("| CUDA ROOT               : ${CUDA_TOOLKIT_ROOT_DIR}"                 )
message("| PHYSICAL CORES          : ${VOXX_PHYSICAL_CORES}"                   )
message("| LOGICAL CORES           : ${VOXX_LOGICAL_CORES}"                    )
message("| CACHELINE SIZE          : ${VOXX_CACHELINE_SIZE}"                   )
message("| CPU INTRINSICS          : ${VOXX_INTRINSICS}"                       )
message("|==---------------------==|\n")
