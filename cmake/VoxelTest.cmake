#==--- CMake/VoxelTest.cmake ------------------------------------------------==#
#
#                                     Voxel
#
#                         Copyright (c) 2017 Rob Clucas
#  
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#
#
# Description :   This file defines a function for building a custom test
#                 module which may include cude code. This function allows
#                 clang to be used to compile the cude code.
#           
#==--------------------------------------------------------------------------==#


#==--------------------------------------------------------------------------==#
# Description             : This function will create a new build target. The
#                           following global parameters are used:
#
# VOXX_TEST_NAME          : The name of the target test.
# VOXX_TEST_FILE          : The absolute path to the test file.
# VOXX_TEST_FLAGS         : Compiler flags specific to the test.
# VOXX_TEST_INCLUDE_DIRS  : Header search directories.
# VOXX_TEST_LIBRARY_DIRS  : Library search directories for linking.
# VOXX_TEST_LIBS          : Library to link with the test.
# VOXX_TEST_DEPENDENCIES  : Any sources which are a dependecy of the test file.
# VOXX_TEST_BINARY_DIR    : Directory to install tests to.
#==--------------------------------------------------------------------------==#
function(voxel_test VOXX_TEST_NAME)
  # Append -I to all include directories.
  foreach(ARG ${VOXX_TEST_INCLUDE_DIRS})
    set(TEST_INCLUDE_DIRS "${TEST_INCLUDE_DIRS} -I${ARG}")   
  endforeach()

  # Append -L to all library directories
  foreach(ARG ${VOXX_TEST_LIBRARY_DIRS})
    set(TEST_LIBRARY_DIRS "${TEST_LIBRARY_DIRS} -L${ARG}")
  endforeach()

  separate_arguments(VOXX_TEST_FLAGS)
  separate_arguments(VOXX_TEST_LIBS)
  separate_arguments(TEST_INCLUDE_DIRS)
  separate_arguments(TEST_LIBRARY_DIRS)
  separate_arguments(CMAKE_CXX_FLAGS)

  # Compile object files for each of the dependencies
  foreach(FILE ${VOXX_TEST_DEPENDENCIES})
    get_filename_component(DEPENDENCY_NAME ${FILE} NAME_WE)
    set(OBJECTS "${OBJECTS} ${DEPENDENCY_NAME}.o")
    add_custom_command(
      OUTPUT  ${DEPENDENCY_NAME}.o
      COMMAND ${CMAKE_CXX_COMPILER}
      ARGS    ${TEST_INCLUDE_DIRS}
              ${CMAKE_CXX_FLAGS}
              ${VOXX_TEST_FLAGS}
              -c ${FILE}
              -o ${DEPENDENCY_NAME}.o
    )
  endforeach()
  separate_arguments(OBJECTS)

  # Compile objct file for test file:
  get_filename_component(TEST_NAME ${VOXX_TEST_FILE} NAME_WE)
  set(OBJECT ${TEST_NAME}.o)
  add_custom_command(
    OUTPUT  ${TEST_NAME}.o
    COMMAND ${CMAKE_CXX_COMPILER}
    ARGS    ${TEST_INCLUDE_DIRS}
            ${CMAKE_CXX_FLAGS}
            ${VOXX_TEST_FLAGS}
            -c ${VOXX_TEST_FILE}
            -o ${TEST_NAME}.o
  )

  # Create a target for the test:
  add_custom_target(
    ${VOXX_TEST_NAME} ALL
    COMMAND ${CMAKE_CXX_COMPILER}
            ${TEST_INCLUDE_DIRS}
            ${VOXX_TEST_FLAGS}
            ${CMAKE_CXX_FLAGS}
            -o ${VOXX_TEST_NAME} ${OBJECT} ${OBJECTS}
            ${TEST_LIBRARY_DIRS}
            ${VOXX_TEST_LIBS}
    DEPENDS ${OBJECT} ${OBJECTS}
  )

  install(
    PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${VOXX_TEST_NAME}
    DESTINATION ${VOXX_TEST_BINARY_DIR}
  )
endfunction()