#==--- cmake/VoxelConfig.cmake ----------------------------------------------==#
#
#                                 	 Voxel 
#
#                         Copyright (c) 2017 Rob Clucas
#  
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#
#
# Description : This file defines the cmake configuration file for voxel.
#           
#==--------------------------------------------------------------------------==#

get_filename_component(Voxel_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)

# Define the cmake installation directory:
set(Voxel_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Provide all the library targets:
include("${CMAKE_CURRENT_LIST_DIR}/VoxelTargets.cmake")

# Include all the custom cmake scripts:
include("${CMAKE_CURRENT_LIST_DIR}/VoxelSystemInfo.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/VoxelTest.cmake")

# Define the include directories:
set(Voxel_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(Voxel_LIBRARY_DIRS "${CMAKE_CURRENT_LIST_DIR}../../../lib"     )
set(Voxel_LIBRARYS)
set(Voxel_DEFINITIONS  -std=c++1z)

set(SupportedComponents SystemInfo)

set(Voxel_FOUND True)

# Check that all the components are found:
# And add the components to the Voxel_LIBS parameter:
foreach(comp ${Voxel_FIND_COMPONENTS})
	if (NOT ";${SupportedComponents};" MATCHES comp)
		set(Voxel_FOUND False)
		set(Voxel_NOT_FOUND_MESSAGE "Unsupported component: ${comp}")
	endif()
	set(Voxel_LIBRARYS "${Voxel_LIBRARYS} -lVoxel{comp}")
	if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/Voxel${comp}Targets.cmake")
		include("${CMAKE_CURRENT_LIST_DIR}/Voxel${comp}Targets.cmake")
	endif()
endforeach()

set(Voxel_LIBS ${Voxel_LIBRARIES})