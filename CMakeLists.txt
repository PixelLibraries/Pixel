#==--- Pixel/CMakeLists.txt ------------------------------------------------==#

CMAKE_MINIMUM_REQUIRED(VERSION 3.4)
PROJECT(Pixel)

#==--- Versioning ----------------------------------------------------------==#

SET(Pixel_MAJOR_VERSION 0)
SET(Pixel_MINOR_VERSION 1)
SET(Pixel_PATCH_VERSION 0)
SET(Pixel_VERSION
  ${Pixel_MAJOR_VERSION}.${Pixel_MINOR_VERSION}.${Pixel_PATCH_VERSION}
)

#==--- Build Type ----------------------------------------------------------==#

IF(NOT CMAKE_BUILD_TYPE) 
  SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type not specified!")
ENDIF(NOT CMAKE_BUILD_TYPE)

#==--- Pixel Root ----------------------------------------------------------==#

IF(WIN32)
  SET(Pixel_ROOT )
ELSE()
  SET(Pixel_ROOT /opt/pixel)
ENDIF(WIN32)

#==--- Commandline Options -------------------------------------------------==#

OPTION(Pixel_HEADER_INSTALL_DIR  
  "Pixel header installation directory" ${Pixel_ROOT}/include/Pixel/)
OPTION(Pixel_LIB_INSTALL_DIR  
  "Pixel library installation directory" ${Pixel_ROOT}/lib/)

#==--- Include directories -------------------------------------------------==#

INCLUDE_DIRECTORIES(${Pixel_SOURCE_DIR}/Include 
                    ${Pixel_ROOT}/include       )

#==--- Subdirectories ------------------------------------------------------==#

ADD_SUBDIRECTORY(Tests)
ADD_SUBDIRECTORY(Docs)

#==--- Compiler Flags ------------------------------------------------------==#

IF(WIN32)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
  SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} /O2"  )
ELSEIF(APPLE)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++1z"     )
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pedantic -Wall")
  SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O3 -std=c++1z"   )
  SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -pedantic -Wall"  )
ELSE()
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++1z"     )
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pedantic -Wall")
  SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O3 -std=c++1z"   )
  SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -pedantic -Wall"  )
ENDIF()

#==--- Asm Flags -----------------------------------------------------------==#

IF(GENERATE_ASM) 
  SET(ASM_FLAGS_REL "-g -Wa,-anhlds -fverbose-asm -masm=intel" CACHE 
    STRING "Assembly release flags" FORCE)
  SET(ASM_FLAGS_DBG "-O3 -Wa,-anhlds -fverbose-asm -masm=intel" CACHE 
    STRING "Assembly debug flags" FORCE)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ASM_FLAGS_REL}")
  SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${ASM_FLAGS_DBG}"  )
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

INSTALL(DIRECTORY ${Pixel_SOURCE_DIR}/Include/Pixel
        DESTINATION ${Pixel_HEADER_INSTALL_DIR})

#==--- Print Configuration -------------------------------------------------==#

MESSAGE("|==--- CONFIGURATION --------------------------------------------==|")
MESSAGE("| PIXEL VERSION           : ${Pixel_VERSION}"                        )
MESSAGE("| BUILD TYPE              : ${CMAKE_BUILD_TYPE}"                     )
MESSAGE("| CMAKE_CXX_FLAGS_RELEASE : ${CMAKE_CXX_FLAGS_RELEASE}"              )
MESSAGE("| CMAKE_CXX_FLAGS_DEBUG   : ${CMAKE_CXX_FLAGS_DEBUG}"                )
MESSAGE("| CMAKE_CXX_FLAGS         : ${CMAKE_CXX_FLAGS}"                      )
MESSAGE("| HEADER INSTALL DIR      : ${Pixel_HEADER_INSTALL_DIR}"             )
MESSAGE("| LIB INSTALL DIR         : ${Pixel_LIB_INSTALL_DIR}"                )
MESSAGE("| GENERATE ASSEMBLY       : ${GENERATE_ASM}"                         )
MESSAGE("| TESTS                   : ${TESTS_STRING}"                         )
MESSAGE("|==--------------------------------------------------------------==|")
