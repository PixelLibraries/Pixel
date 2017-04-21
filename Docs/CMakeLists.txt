#==--- Utility/docs/CMakeLists.txt ------------------------------------------==#

find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile 
    @ONLY)

  add_custom_target(Docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
    COMMENT "Generate doxygen documentation")
endif(DOXYGEN_FOUND)