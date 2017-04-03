#==--- Pixel/Docs/CMakeLists.txt -------------------------------------------==#

FIND_PACKAGE(Doxygen QUIET)

IF (DOXYGEN_FOUND)
  CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile 
    @ONLY
  )

  ADD_CUSTOM_TARGET(Docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
    COMMENT "Generate doxygen documentation"
  )

  INSTALL(DIRECTORY   ${CMAKE_CURRENT_BINARY_DIR}/html 
          DESTINATION ${Pulley_SOURCE_DIR}/Bin/Docs/)
ENDIF(DOXYGEN_FOUND)

#==-------------------------------------------------------------------------==#

