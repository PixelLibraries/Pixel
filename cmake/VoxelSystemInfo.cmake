#==--- CMake/VoxelSystemInfo.cmake ------------------------------------------==#
#
#                                     Voxel 
#
#                         Copyright (c) 2017 Rob Clucas
#  
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#
#
# Description	: This file defines a function which gets system information
#					      for the hardware and adds appropriate definitions to the 
#					      C++ source code which allow the information to be used at
#					      compile time.
# 					
#==--------------------------------------------------------------------------==#

#==--------------------------------------------------------------------------==#
# Description : This function will find system hardware information
#							  and add appropriate definitions to the C++ source
#							  code. The following definitions are defined:
#
#==--------------------------------------------------------------------------==#

function(voxel_system_info)
  set(SYS_INFO "")
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
                        /opt/Voxel
                        /usr/local/Voxel)
  find_package(Voxel)

  # If this is the Utility project, then it's likely that Voxel will not be
  # installed, and therefore some of the applications aren't. In this case,
  # we build them dynamically so that we can run them from the build script
  # to generate the appropriate data:
  if (${CMAKE_PROJECT_NAME} MATCHES "Voxel")
    # Ensures that we don't get a non-terminating recursive build:
    if (${GENERATE_SYSTEM_INFO})
      if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/build)
        execute_process(
          COMMAND rm -rf build
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          OUTPUT_QUIET)
      endif()

      message("-- Generating system information")

      execute_process(
        COMMAND mkdir build
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        OUTPUT_QUIET)

      # Create a new build directory, and run cmake again to create the
      # SystemInformation application:
      execute_process(
        COMMAND           cmake -DCMAKE_BUILD_TYPE=Release
                                -DGENERATE_SYSTEM_INFO=OFF ../../
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build
        OUTPUT_QUIET
        ERROR_QUIET)

      # Make the SystemInformation application:
      execute_process(
        COMMAND           make SystemInformation
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build
        OUTPUT_QUIET)

      # Run the SystemInformation application to generate the system info data
      # from which the definitions can be added into the source code:
      execute_process(
        COMMAND           ./SystemInformation
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build/apps
        OUTPUT_VARIABLE   SYSTEM_INFO)
        set(SYSTEM_INFO ${SYSTEM_INFO} PARENT_SCOPE)

      message("-- Generating system information - done")
    endif()

    # Propogate the SYSTEM_INFO variable out of this scope:
    set(SYSTEM_INFO ${SYSTEM_INFO} PARENT_SCOPE)

  # This branch is taken when Voxel is already installed:
  # TODO: Change this to find_package(Voxel)
  elseif(Voxel_FOUND)
    execute_process(
      COMMAND         ${Voxel_DIR}/bin/SystemInformation
      OUTPUT_VARIABLE SYSTEM_INFO
      OUTPUT_QUIET)
    set(SYSTEM_INFO ${SYSTEM_INFO} PARENT_SCOPE)
  endif()

  if(NOT ${SYSTEM_INFO}  MATCHES ".*[a-zA-z].*")
    if (${GENERATE_SYSTEM_INFO})
      # SystemInformation could not be built =( this should never happen!
      message("\nError:"                                                ) 
      message("\n        The SystemInformation executable could not be" )
      message("         found in ${Voxel_ROOT}/bin. If building the"    )
      message("         Utility library, this should never happen, and" )
      message("         if building another library, this is likely"    )
      message("         because Voxel_ROOT is not found, due to the"    )
      message("         Utility library not being installed. Please"    )
      message("         install it and try again :)\n"                  )
    endif()

  # This branch is taken when we have been able to generate valid system
  # information (which should be always):
  else()
    # Convert to list:
    string(REPLACE "\r\n" "; " SYS_INFO ${SYSTEM_INFO})
    string(REPLACE "\n"   "; " SYS_INFO ${SYSTEM_INFO})

    foreach(PARAM ${SYS_INFO})
      if(${PARAM} MATCHES "Physical CPU.*")
        string(REGEX MATCH "[0-9]+" CPUS ${PARAM})
        set(VOXX_CPU_COUNT ${CPUS} PARENT_SCOPE)
        add_definitions(-DVoxxCpuCount=${CPUS})
      elseif(${PARAM} MATCHES "Physical Cores[^/].*")
        string(REGEX MATCH "[0-9]+" CORES ${PARAM})
        set(VOXX_PHYSICAL_CORES ${CORES} PARENT_SCOPE)
        add_definitions(-DVoxxPhysicalCores=${CORES})
      elseif(${PARAM} MATCHES "Logical Cores[^/].*")
        string(REGEX MATCH "[0-9]+" CORES ${PARAM})
        set(VOXX_LOGICAL_CORES ${CORES} PARENT_SCOPE)
        add_definitions(-DVoxxLogicalCores=${CORES})
      elseif(${PARAM} MATCHES "Cacheline Size.*")
        string(REGEX MATCH "[0-9]+" SIZE ${PARAM})
        set(VOXX_CACHELINE_SIZE ${SIZE} PARENT_SCOPE)
        add_definitions(-DVoxxCachelineSize=${SIZE})
      elseif(${PARAM} MATCHES "L1 Cache.*")
        string(REGEX MATCH " [0-9]+" L1SIZE ${PARAM})
        add_definitions(-DVoxxL1CacheSize=${L1SIZE})
      elseif(${PARAM} MATCHES "L2 Cache.*")
        string(REGEX MATCH " [0-9]+" L2SIZE ${PARAM})
        add_definitions(-DVoxxL2CacheSize=${L2SIZE})
      elseif(${PARAM} MATCHES "L3 Cache.*")
        string(REGEX MATCH " [0-9]+" L3SIZE ${PARAM})
        add_definitions(-DVoxxL3CacheSize=${L3SIZE})
      elseif(${PARAM} MATCHES "L1 Sharing.*")
        string(REGEX MATCH " [0-9]+" L1SHARING ${PARAM})
        add_definitions(-DVoxxL1Sharing=${L1SHARING})
      elseif(${PARAM} MATCHES "L2 Sharing.*")
        string(REGEX MATCH " [0-9]+" L2SHARING ${PARAM})
        add_definitions(-DVoxxL2Sharing=${L2SHARING})
      endif()
    endforeach()
  endif()
endfunction(voxel_system_info)