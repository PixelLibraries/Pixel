# Voxel

This is the main repository for Voxel software. It contains general purpose
functionality that is used in other, more specific, Voxel libraries. Most code in
this library is code which was initially implemented for a specific purpose, but
was then brought here as it's likely useful in multiple places.

## Installation

Voxel uses Cmake for installation. Some of the applications in the ```apps/```
directory are used to generate output which can be parsed by Cmake to generate
definitions which are used to generate ```constexpr``` functions. If the Voxel is
not already installed, then the application binaries will not be found. When
installing Voxel for the first time, the installation script compiles
the ```SystemInformation``` application so that the information can be added into the C++ source code.

### Linux or OSX

The following are the most basic options for installation:

| Variable             | Description                       | Options       |
|:---------------------|:----------------------------------|:--------------|
| CMAKE_BUILD_TYPE     | The build type                    | Debug/Release |
| CMAKE_INSTALL_PREFIX | Path to install directory         | User defined  |
| BUILD_SHARED_LIBS    | Build libraries as shared         | ON/NONE	   |

__Note:__ The build script automatically appends ```Voxel``` to
	        to ```CMAKE_INSTALL_PREFIX``` if it __is not__ part of the variable. So ```-DCMAKE_INSTALL_PREFIX=/opt``` will install all software
          to ```/opt/Voxel```. If ```Voxel``` __is__ found in the
	        the ```CMAKE_INSTALL_PREFIX``` variable then the build script
          __does not__ modify the variable, and it is used __as is__. For
          example, ```-DCMAKE_INSTALL_PREFIX=/opt```
          and ```-DCMAKE_INSTALL_PREFIX=/opt/Voxel```will both install
          to ```/opt/Voxel```.

To build Voxel with __static__ libraries:
~~~py
# From the root directory:
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release  \
      -DCMAKE_INSTALL_PREFIX=/opt \
      ..
sudo make install
~~~

To build Voxel with __shared__ libraries:
~~~py
# From the root directory:
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release  \
      -DCMAKE_INSTALL_PREFIX=/opt \
      -DBUILD_SHARED_LIBS=BOOL:ON \
      ..
sudo make install
~~~

Which will install the library to ```/opt/Voxel```, after which 
```make {component}``` will make any of the components.

## Components

Components are the header files and built libraries that the part of the Voxel
repository. Using them is as simple as including the relevant header from
```include/Voxel/Component``` and linking against the library, if there is one.

### Find Package for Voxel

When installing Voxel, a VoxelConfig.cmake file is generated and installed in
```CMAKE_INSTALL_PREFIX/lib/cmake/Voxel```, which allows other libraries to use
Voxel. To ensure that the VoxelConfig.cmake package is found, add the Voxel
intallation path to the ```CMAKE_PREFIX_PATH``` variable when using Voxel. In
the CMakeLists.txt file, add the following:

~~~cmake
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} VOXEL_INSTALL_PREFIX")
find_package(Voxel)
~~~

to find Voxel, where ```VOXEL_INSTALL_PREFIX``` is the installation prefix for
Voxel. The ```find_package``` script defines the following Cmake variables:

| Variable             | Description                                              |
|:---------------------|:---------------------------------------------------------|
| Voxel_INCLUDES       | The include directories for compilation                  |
| Voxel_LIBRARIES      | All of the voxel libraries to link                       |
| Voxel_LIBS           | Only libraries corresponding to those used with COMPONENTS with cmake ```find_package``` |
| Voxel_DEFINITIONS    | Required compiler definitions for Voxel                  |

### Linking

Assuming that ```find_packge(Voxel ...)``` has been run, to link against a
single library from cmake, simple do (see list of components below):

~~~
target_link_libraries(Target Voxx::VoxelComponent)
~~~

where ```VoxelComponent``` is the component to link against. Alternatively,
to link against numerous components, list the components when using
```find_package```:

~~~
find_package(Voxel COMPONENTS SystemInfo)
~~~

which populates the ```Voxel_LIBS``` cmake variable with the appropriate library
definitions. A target can then be linked as follows:

~~~
target_link_libraries(Target ${Voxel_LIBS})
~~~

Finally, to link against all the Voxel libraries, use ```Voxel_LIBRARIES```:

~~~
target_link_libraries(Target ${Voxel_LIRARIES})
~~~

### System Info

The system information component (at ```include/Voxel/SystemInfo/```)
defines cross-platform functionality to get information for various components of
the system (such as cpu info, gpu info, etc). The component can be linked with
```Voxx::VoxelSystemInfo```, or by specifying the component with
```find_package```, i.e:

~~~
# Find only SystemInfo:
find_package(Voxel COMPONENTS SystemInfo)
...

# Link target with SystemInfo:
target_link_libraries(Target Voxx::VoxelSystemInfo)
~~~

## Applications

Voxel provides numerous applications as binaries, they can be built using
~~~sh
make ApplicationName
~~~
after running Cmake as shown above.

### System Information

The system information component application (```bin/SystemInformation``` after
installation) displays the system information to the console.

## Licensing

This librarry is licensed under the MIT license, and is completely free -- you may do whatever you like with it!

