#==--- cmake/VoxelAddExecutable.cmake ---------------------------------------==#
#
#                                     Voxel
#
#                         Copyright (c) 2017 Rob Clucas
#  
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#
#
# Description :   This file defines functions to create executables which can
#                 support compiling source files such that execution can occur
#                 on both the CPU and the GPU using cuda.
#           
#==--------------------------------------------------------------------------==#

#==--------------------------------------------------------------------------==#
# Description             : This function will add include directories to the
#                           list of directories used by voxx_add_executable,
#                           much like include_directories adds directories to 
#                           add_executable.
#                           
#                           This will only set the directories specified by
#                           VOXX_DIRECTORIES to be included, if you want to
#                           append to the current list, then see
#                           voxx_include_directories_append.
#
# VOXX_DIRECTORIES        : A list of directories to include.
#==--------------------------------------------------------------------------==#
function(voxx_include_directories VOXX_DIRECTORIE)
  set(VOXX_INCLUDE_DIRS "${VOXX_DIRECTORIES} ${ARGN}"
      CACHE FORCE "voxx include directories" FORCE)
endfunction()

#==--------------------------------------------------------------------------==#
# Description             : This function will append directories to the
#                           list of directories used by voxx_add_executable,
#                           much like include_directories adds directories to 
#                           add_executable.
#                           
# VOXX_DIRECTORIES        : A list of directories to append to the current list.
#==--------------------------------------------------------------------------==#
function(voxx_include_directories_append VOXX_DIRECTORIES)
  set(VOXX_INCLUDE_DIRS "${VOXX_INCLUDE_DIRS} ${VOXX_DIRECTORIES} ${ARGN}"
      CACHE FORCE "voxx include directories" FORCE)
endfunction()

#==--------------------------------------------------------------------------==#
# Description             : This function set the list of directories to search
#                           for linking.
#                           
# VOXX_DIRECTORIES        : A list of directories to append to the current list.
#==--------------------------------------------------------------------------==#
function(voxx_library_directories VOXX_DIRECTORIES)
  set(VOXX_LIBRARY_DIRS "${VOXX_DIRECTORIES} ${ARGN}"
      CACHE FORCE "voxel library directories" FORCE)
endfunction()

#==--------------------------------------------------------------------------==#
# Description             : This function adds definitions for all targets.
#                           
# VOXX_DEFINITIONS        : The libraries to link against.
#==--------------------------------------------------------------------------==#
function(voxx_add_definitions VOXX_DEFINITIONS)
  set(VOXX_GLOBAL_DEFINITIONS "${VOXX_DEFINITIONS} ${ARGN}"
      CACHE FORCE "voxel definitions" FORCE)
endfunction()

#==--------------------------------------------------------------------------==#
# Description             : This function set compiler flags specifically for 
#                           the target.
#                           
# VOXX_TARGET             : The target to set the link directories for.
# VOXX_FLAGS              : The libraries to link against.
#==--------------------------------------------------------------------------==#
function(voxx_target_flags VOXX_TARGET VOXX_FLAGS)
  set(${VOXX_TARGET}_FLAGS "${VOXX_FLAGS} ${ARGN}"
      CACHE FORCE "voxel target: ${VOXX_TARGET}_FLAGS" FORCE)
endfunction()

#==--------------------------------------------------------------------------==#
# Description             : This function will set the list of librarys for
#                           which a target must be linked against.
#                           
# VOXX_TARGET             : The target to set the link directories for.
# VOXX_LINK_LIBS          : The libraries to link against.
#==--------------------------------------------------------------------------==#
function(voxx_target_link_libraries VOXX_TARGET VOXX_LINK_LIBS)
  set(${VOXX_TARGET}_LINK_LIBS "${VOXX_LINK_LIBS} ${ARGN}"
      CACHE FORCE "voxel link libraries: ${VOXX_TARGET}_LINK_LIBS" FORCE)
endfunction()

#==--------------------------------------------------------------------------==#
# Description             : This function will create a new build target. The
#                           following global parameters are used:
#
# VOXX_TARGET             : The name of the target.
# VOXX_TARGET_FILE        : The source file assosciated with the target.
#
#==--------------------------------------------------------------------------==#
function(voxx_add_executable VOXX_TARGET VOXX_TARGET_FILE)
  set(VOXX_TARGET_LIST "${VOXX_TARGET_LIST} ${VOXX_TARGET}"
      CACHE FORCE "voxel target list" FORCE)

  set(${VOXX_TARGET}_FILE "${VOXX_TARGET_FILE}"
      CACHE FORCE "target file" FORCE)
  set(${VOXX_TARGET}_DEPENDENCIES "${ARGN}"
      CACHE FORCE "dependecies" FORCE)
endfunction()


#==--------------------------------------------------------------------------==#
# Description             : This function will create all the targets which have
#                           been added with voxx_add_executable. It must be
#                           called at the end of the root CMakeList.txt file.
#==--------------------------------------------------------------------------==#
function(voxx_create_all_targets)
  # Append -I to all include directories.
  foreach(ARG ${VOXX_INCLUDE_DIRS})
    set(TARGET_INCLUDE_DIRS "${TARGET_INCLUDE_DIRS} -I${ARG}")
  endforeach()

  # Append -L to all library directories
  foreach(ARG ${VOXX_LIBRARY_DIRS})
    set(TARGET_LIBRARY_DIRS "${TARGET_LIBRARY_DIRS} -L${ARG}")
  endforeach()

  if (VOXX_GLOBAL_DEFINITIONS)
    separate_arguments(VOXX_GLOBAL_DEFINITIONS)
  endif()
  if (CMAKE_CXX_FLAGS)
    separate_arguments(CMAKE_CXX_FLAGS)
  endif()
  if (TARGET_INCLUDE_DIRS)
    separate_arguments(TARGET_INCLUDE_DIRS)
  endif()
  if(TARGET_LIBRARY_DIRS)
    separate_arguments(TARGET_LIBRARY_DIRS)
  endif()

  separate_arguments(VOXX_TARGET_LIST)
  foreach(VOXX_TARGET ${VOXX_TARGET_LIST})
    string(REGEX REPLACE " " "" VOXX_TARGET ${VOXX_TARGET})
    message("Creating Target -- ${VOXX_TARGET}")

    if (${VOXX_TARGET}_FLAGS)
      separate_arguments(${VOXX_TARGET}_FLAGS)
    endif()
    if (${VOXX_TARGET}_LINK_LIBS)
      separate_arguments(${VOXX_TARGET}_LINK_LIBS)
    endif()

    # Compile object files for each of the dependencies
    foreach(FILE ${${VOXX_TARGET}_DEPENDENCIES})
      get_filename_component(DEPENDENCY_NAME ${FILE} NAME_WE)
      set(OBJECTS "${OBJECTS} ${DEPENDENCY_NAME}.o")
      add_custom_command(
        OUTPUT  ${DEPENDENCY_NAME}.o
        COMMAND ${CMAKE_CXX_COMPILER}
        ARGS    ${TARGET_INCLUDE_DIRS}
                ${CMAKE_CXX_FLAGS}
                ${VOXX_GLOBAL_DEFINITIONS}
                ${${VOXX_TARGET}_FLAGS}
                -c ${FILE}
                -o ${DEPENDENCY_NAME}.o)
    endforeach()
    separate_arguments(OBJECTS)

    # Compile object file for test file:
    get_filename_component(TARGET_NAME ${${VOXX_TARGET}_FILE} NAME_WE)
    set(OBJECT ${TARGET_NAME}.o)
    add_custom_command(
      OUTPUT  ${TARGET_NAME}.o
      COMMAND ${CMAKE_CXX_COMPILER}
      ARGS    ${TARGET_INCLUDE_DIRS}
              ${CMAKE_CXX_FLAGS}
              ${VOXX_GLOBAL_DEFINITIONS}
              ${${VOXX_TARGET}_FLAGS}
              -c ${${VOXX_TARGET}_FILE}
              -o ${TARGET_NAME}.o)

    # Create a target for the test:
    #string(REGEX REPLACE " " "" THIS_TARGET ${VOXX_TARGET})
    add_custom_target(
      ${VOXX_TARGET} ALL
      COMMAND ${CMAKE_CXX_COMPILER}
              ${TARGET_INCLUDE_DIRS}
              ${CMAKE_CXX_FLAGS}
              ${${VOXX_TARGET}_FLAGS}
              ${VOXX_GLOBAL_DEFINITIONS}
              -o ${VOXX_TARGET} ${OBJECT} ${OBJECTS}
              ${TARGET_LIBRARY_DIRS}
              ${${VOXX_TARGET}_LINK_LIBS}
      DEPENDS ${OBJECT} ${OBJECTS})

    install(
      PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${VOXX_TARGET}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
    set(OBJECT)
    set(OBJECTS)
    message("Created Target  -- ${VOXX_TARGET}")

    # Clean up:
    set(${VOXX_TARGET}_DEPENDENCIES "" CACHE FORCE "")
    set(${VOXX_TARGET}_FILE         "" CACHE FORCE "")
  endforeach()
  set(VOXX_TARGET_LIST "" CACHE FORCE "voxel target list" FORCE)
endfunction()