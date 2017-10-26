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
include("${CMAKE_CURRENT_LIST_DIR}/VoxelAddExecutable.cmake")

find_package(CUDA)
if (CUDA_FOUND)
  if(APPLE)
    set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    set(CUDA_LIBS  -lcudart_static -ldl -pthread      )
  else()
    set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    set(CUDA_LIBS  -lcudart_static -ldl -lrt -pthread   )
  endif()

  set(Voxel_CUDA_SUPPORT TRUE)
  set(Voxel_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

  # Change this to find SM version.
  set(Voxel_DEFINITIONS  -DVoxxCudaSupported
                         --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
  set(Voxel_LIBRARY_DIRS ${CUDA_LIBRARY_DIRS})
  set(Voxel_LIBRARIES    ${CUDA_LIBS})
  set(Voxel_LIBS         ${CUDA_LIBS})
endif()

set(VOXX_LIBS "-lHeterogeneous -lSystemInfo -lThread")
set(VOXX_LDIR "${CMAKE_CURRENT_LIST_DIR}/../../../lib")

# Define the include directories:
set(Voxel_INCLUDE_DIRS ${Voxel_INCLUDE_DIRS}
                       ${CMAKE_CURRENT_LIST_DIR}/../../../include      )
set(Voxel_LIBRARY_DIRS "${Voxel_LIBRARY_DIRS} ${VOXX_LDIR}"            )
set(Voxel_LIBRARIES    "${Voxel_LIBRARIES} ${VOXX_LIBS}"               )
set(Voxel_DEFINITIONS  "${Voxel_DEFINITIONS} -std=c++17 -stdlib=libc++")

set(SupportedComponents Heterogeneous SystemInfo Thread)

set(Voxel_FOUND True)

# Check that all the components are found:
# And add the components to the Voxel_LIBS parameter:
foreach(comp ${Voxel_FIND_COMPONENTS})
	if (NOT ";${SupportedComponents};" MATCHES comp)
		set(Voxel_FOUND False)
		set(Voxel_NOT_FOUND_MESSAGE "Unsupported component: ${comp}")
	endif()
	set(Voxel_LIBS "${Voxel_LIBS} -l{comp}")
	if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${comp}Targets.cmake")
		include("${CMAKE_CURRENT_LIST_DIR}/${comp}Targets.cmake")
	endif()
endforeach()