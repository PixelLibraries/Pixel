#==--- Utility/src/CMakeLists.txt -------------------------------------------==#

include_directories(${CUDA_INCLUDE_DIRS} ${Voxel_SOURCE_DIR}/include)

#==--- Libraries ------------------------------------------------------------==#

add_library(Heterogeneous
            ${Voxel_SOURCE_DIR}/src/Heterogeneous/HeteroException.cpp)
install(TARGETS  Heterogeneous
        EXPORT   VoxelLibs
        ARCHIVE  DESTINATION lib
        LIBRARY  DESTINATION lib
        RUNTIME  DESTINATION bin)
target_include_directories(Heterogeneous PUBLIC ${CMAKE_INSTALL_PREFIX}/include)
target_link_libraries(Heterogeneous -lc++abi -lc++experimental)

add_library(SystemInfo ${Voxel_SOURCE_DIR}/src/System/SystemInfo.cpp)
install(TARGETS  SystemInfo
        EXPORT   VoxelLibs
        ARCHIVE  DESTINATION lib
        LIBRARY  DESTINATION lib
        RUNTIME  DESTINATION bin)
target_include_directories(SystemInfo PUBLIC ${CMAKE_INSTALL_PREFIX}/include)
target_link_libraries(SystemInfo -lc++abi -lc++experimental)

add_library(Thread ${Voxel_SOURCE_DIR}/src/Thread/Thread.cpp)
install(TARGETS  Thread
        EXPORT   VoxelLibs
        ARCHIVE  DESTINATION lib
        LIBRARY  DESTINATION lib
        RUNTIME  DESTINATION bin)
target_include_directories(Thread PUBLIC ${CMAKE_INSTALL_PREFIX}/include)
target_link_libraries(Thread -lc++abi -lc++experimental)
