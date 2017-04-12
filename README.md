# Utility

This library contains general purpose functionality that is used in other Voxel libraries. Most code in this library is code which was initially implemented
for a specific purpose, but was then brought here as it's likely useful in
multiple places.

## Installation

Utility uses Cmake for installation. Some of the applications in the ```Apps/```
directory are used to generate output which can be parsed by Cmake to generate
definitions which are used to generate ```constexpr``` functions. If the Utility
library has not been installed, then the application binaries will not be found,
thus, when installing Utility for the first time, some of the commands need to be
run twice (see below).

### Linux or OSX

The following are the most basic options for installation:

|--------------------------------------------------------------------------|
| Variable             | Description                       | Options       |
|:--------------------:|:---------------------------------:|:-------------:|
| CMAKE_BUILD_TYPE     | The build type                    | Debug/Release |
| CMAKE_INSTALL_PREFIX | Path to install directory         | User defined  |
| VOXX_ROOT 		   | Root directory for Voxel software | User defined  |
|--------------------------------------------------------------------------|

__Note:__ The build scripts automatically append ```Voxel``` to
		  CMAKE_INSTALL_PREFIX, so ```/opt``` will install all software to
		  ```/opt/Voxel```.

```
# From the root directory:
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release
	  -DCMAKE_INSTALL_PREFIX=/opt
	  -DVOXX_ROOT=/opt/Voxel ..
sudo make install
cmake ..
sudo make install
```

Which will install the library to ```/opt/Voxel```, after which 
```make {component}``` will make any of the components, and
```sudo make install``` will install any of the built components which are not
installed by default.

## Components

### System Information

The system information component (at ```Include/Voxel/Utility/SystemInfo```)
defines cross-platform functionality to get information for various components of
the system (such as cpu info, gpu info, etc). 

The application which prints the system information is at
```Apps/SystemInformation.cpp``` and can be built with
```make SystemInformation``` after running cmake (it is installed by default when
using the commands shown in [installation](#installation)).

Some of the functionality is ```constexpr``` and can therefore be used at compile
time. This is done by running the ```SystemInformation``` program from Cmake,
parsing the output, and feeding the results in as definitions, which are wrapped
in ```constexpr``` functions (see ```CpuInfo.hpp``` for the available functions).


## Licensing

This librarry is licensed under the MIT license, and is completely free -- you may do whatever you like with it!

