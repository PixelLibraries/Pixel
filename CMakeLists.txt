#==--- Utility/CMakeLists.txt ------------------------------------------------==#

CMAKE_MINIMUM_REQUIRED(VERSION 3.4)
PROJECT(Utility)

#==--- Versioning ----------------------------------------------------------==#

SET(Utility_MAJOR_VERSION 0)
SET(Utility_MINOR_VERSION 1)
SET(Utility_PATCH_VERSION 0)
SET(Utility_VERSION
  ${Utility_MAJOR_VERSION}.${Utility_MINOR_VERSION}.${Utility_PATCH_VERSION}
)

#==--- Build Type ----------------------------------------------------------==#

IF(NOT CMAKE_BUILD_TYPE) 
  SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type not specified!")
ENDIF(NOT CMAKE_BUILD_TYPE)

#==--- Voxel Root ----------------------------------------------------------==#

IF(WIN32)
  SET(Voxel_ROOT )
ELSE()
  SET(Voxel_ROOT /opt/voxel)
ENDIF(WIN32)

#==--- Commandline Options -------------------------------------------------==#

OPTION(Utility_HEADER_INSTALL_DIR  "Utility header installation directory"
       ${Voxel_ROOT}/include/Voxel/)
OPTION(Utility_LIB_INSTALL_DIR  "Utility library installation directory"
       ${Voxel_ROOT}/lib/)

#==--- Include directories -------------------------------------------------==#

INCLUDE_DIRECTORIES(${Utility_SOURCE_DIR}/Include 
                    ${Voxel_ROOT}/include       )

#==--- Subdirectories ------------------------------------------------------==#

ADD_SUBDIRECTORY(Tests)
ADD_SUBDIRECTORY(Docs)

#==--- Compiler Flags ------------------------------------------------------==#

IF(WIN32)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
  SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} /O2"  )
ELSEIF()
  SET(FLAGS "-std=c++1z -O3 -pedantic -Wall -Wextra"               ) 
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${FLAGS}") 
  SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${FLAGS}"  )
ENDIF()

#==--- Set CXXFLAGS --------------------------------------------------------==#

IF(CMAKE_BUILD_TYPE MATCHES Release)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}"
      CACHE STRING "CXX FLAGS" FORCE)
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}"
      CACHE STRING "CXX FLAGS" FORCE)
ENDIF()

#==--- Tests ---------------------------------------------------------------==#

SET(TESTS_STRING)

#==--- Install -------------------------------------------------------------==#

INSTALL(DIRECTORY ${Utility_SOURCE_DIR}/Include/Voxel
        DESTINATION ${Utility_HEADER_INSTALL_DIR})

#==--- Print Configuration -------------------------------------------------==#

MESSAGE("|==--- CONFIGURATION --------------------------------------------==|")
MESSAGE("| UTILITY VERSION         : ${Utility_VERSION}"                      )
MESSAGE("| BUILD TYPE              : ${CMAKE_BUILD_TYPE}"                     )
MESSAGE("| CMAKE_CXX_FLAGS_RELEASE : ${CMAKE_CXX_FLAGS_RELEASE}"              )
MESSAGE("| CMAKE_CXX_FLAGS_DEBUG   : ${CMAKE_CXX_FLAGS_DEBUG}"                )
MESSAGE("| CMAKE_CXX_FLAGS         : ${CMAKE_CXX_FLAGS}"                      )
MESSAGE("| HEADER INSTALL DIR      : ${Utility_HEADER_INSTALL_DIR}"           )
MESSAGE("| LIB INSTALL DIR         : ${Utility_LIB_INSTALL_DIR}"              )
MESSAGE("| GENERATE ASSEMBLY       : ${GENERATE_ASM}"                         )
MESSAGE("| TESTS                   : ${TESTS_STRING}"                         )
MESSAGE("|==--------------------------------------------------------------==|")
