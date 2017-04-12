#==--- CMake/VoxxSystemInfo.cmake -------------------------------------------==#
#
#                                 Voxel : Utility 
#
#                         Copyright (c) 2017 Rob Clucas
#  
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#
#
# Description	: 	This file defines a function which gets system information
#					for the hardware and adds appropriate definitions to the 
#					C++ source code which allow the information to be used at
#					compile time.
#
# 					
#==--------------------------------------------------------------------------==#

#==--------------------------------------------------------------------------==#
# Description 	          : This function will find system hardware information
#							              and add appropriate definitions to the C++ source
#							              code. The following definitions are defined:
#
#==--------------------------------------------------------------------------==#

function(voxx_system_info)
  if(EXISTS ${VOXX_ROOT}/bin/SystemInformation)
    execute_process(
      COMMAND ${VOXX_ROOT}/bin/SystemInformation OUTPUT_VARIABLE SYS_INFO
    )
    
    # Convert to list:
    string(REPLACE "\r\n" "; " SYS_INFO ${SYS_INFO})
    string(REPLACE "\n"   "; " SYS_INFO ${SYS_INFO})

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
  else()
    # SystemInformation has not been built, display message saying install must
    # be run twice.
    message("\nWarning:"                                                ) 
    message("\n        The SystemInformation executable does not exist" )
    message("         in ${VOXX_ROOT}/bin. If this is the first time"   )
    message("         you are building Utility, this is expected. After")
    message("         cmake completes, run:"                            )
    message("         \t make SystemInformation"                        )
    message("         \t sudo make install"                             )
    message("         \t cmake {CMake PARAMs} .."                       )
    message("         to install with constexpr system information."    )
    message("         The build will not fail without this, but the"    )
    message("         constexpr versions of the system information"     )
    message("         functions will not work.\n"                       )
  endif()
endfunction(voxx_system_info)