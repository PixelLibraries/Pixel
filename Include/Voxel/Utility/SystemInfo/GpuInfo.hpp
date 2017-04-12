//==--- Utility/SystemInfo/GpuInfo.hpp --------------------- -*- C++ -*- ---==//
//            
//                                Voxel : Utility 
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  GpuInfo.hpp
/// \brief This file defines functionality to get gpu information.
//
//==------------------------------------------------------------------------==//

#pragma once

#include <Voxel/Utility/Bitwise.hpp>
#include <Voxel/Utility/Debug.hpp>
#include <Voxel/Utility/Portability.hpp>
#include <Voxel/Utility/Io/IoFwd.hpp>

namespace Voxx    {
namespace Utility {
namespace System  {

#if defined(VoxxGpuCount)         // Compile time version

/// Returns the total number of GPUs in the system.
VoxxDeviceHost static constexpr auto gpuCount() -> int {
  return VoxxGpuCount;
}

#else

/// Returns the total number of GPUs in the system.
VoxxDeviceHost static auto gpuCount() -> int {
#if defined(VoxxCudaSupported)
  int deviceCount = 0;
  VoxxCudaErrorCheck(cudaGetDeviceCount(&deviceCount));
  return deviceCount;
#else
  return 0;
#endif // VoxxCudaSupported
}

#endif // VoxxGpuCount

} // namespace System

/// The GpuArch enum defines the supported GPU architectures.
enum GpuArch : uint8_t {
  Sm20 = 0x20, // Fermi   Generation (SM 2.0) GF100 class
  Sm21 = 0x21, // Fermi   Generation (SM 2.1) GF10x class
  Sm30 = 0x30, // Kepler  Generation (SM 3.0) GK10x class
  Sm32 = 0x32, // Kepler  Generation (SM 3.2) GK10x class
  Sm35 = 0x35, // Kepler  Generation (SM 3.5) GK11x class
  Sm37 = 0x37, // Kepler  Generation (SM 3.7) GK21x class
  Sm50 = 0x50, // Maxwell Generation (SM 5.0) GM10x class
  Sm52 = 0x52, // Maxwell Generation (SM 5.2) GM20x class
  Sm53 = 0x53, // Maxwell Generation (SM 5.3) GM20x class
  Sm60 = 0x60, // Pascal  Generation (SM 6.0) GP100 class
  Sm61 = 0x61, // Pascal  Generation (SM 6.1) GP10x class
  Sm62 = 0x62, // Pascal  Generation (SM 6.2) GP10x class
};

/// This struct stores GPU information, and can be set at compile time and thus
/// used in constant expressions and in templates. The class defines the
/// following information related to the GPU:
/// 
/// compute arch    : The compute architecture version.
/// physical cores  : The number of physical cores for the GPU.
/// multiprocessors : The number of multiprocessors.
/// 
/// There are two usages of the class. The first is the runtime version which
/// is created using the default constructor:
/// 
/// \code{.cpp}
/// // Information is found at runtime:
/// // This version is used when the parameter gpuId is less than 2^20 (a
/// // reasonable limit on the upper bound for the number of GPUs):
/// auto gpuInfo = GpuInfo{gpuId}; 
/// \endcode
/// 
/// Alternatively, the a constexpr version can be created if the binary
/// representation of the information is known:
/// 
/// \code{.cpp}
/// constexpr auto gpuInfo = GpuInfo(GpuInfoBinaryValue);
/// \endcode
class GpuInfo {
 public:
  //==--- Con/destructors --------------------------------------------------==//

  /// Constructor -- if \p valueOrIndex is smallish number (<1M), indicating
  /// that it must be used to get the information for a specific GPU with
  /// device number \p valueOrIndex, then it is used as the number of the
  /// device to get the info for, otherwise (when the number is large,
  /// indicating that some of the flags are set) it is treated as the binary
  /// information which can be used at compile time.
  /// 
  /// Using the class as constexpr with valueOrIndex with a value which is
  /// treated as a device number will result in a compiler error.
  /// 
  /// \param[in]  valueOrIndex  The binary value of the gpuinfo or the device
  ///                           number to get the info for. 
  constexpr GpuInfo(uint64_t valueOrIndex) : Value(valueOrIndex) {
    if (!(valueOrIndex & Mask::CoresPerSm))
      fillGpuInfo(valueOrIndex);
  }

  //==--- Getters ----------------------------------------------------------==//
  
  /// Returns the gpu compute architecture version.
  VoxxDeviceHost constexpr auto architecture() const noexcept -> uint64_t {
    return getProperty(Mask::Arch);
  }

  /// Returns the number of streaming multiprocessors.
  VoxxDeviceHost constexpr auto multiprocessors() const noexcept -> uint64_t {
    return getProperty(Mask::Multiprocessors);
  }

  /// Returns the number of physical cores.
  VoxxDeviceHost constexpr auto physicalCores() const noexcept -> uint64_t {
    return getProperty(Mask::CoresPerSm) * getProperty(Mask::Multiprocessors);
  }

  /// Returns the number of cores per SM.
  VoxxDeviceHost constexpr auto
  coresPerMultiProcessor() const noexcept -> uint64_t {
    return getProperty(Mask::CoresPerSm);
  }

  /// Returns the compute architecture version in the form it is used as a
  /// compiler flag : sm_{major}{minor}.
  auto architectureString() const -> std::string {
    const auto major = ((getProperty(Mask::Arch) >> 4) & 0x0F) * 10;
    const auto minor = getProperty(Mask::Arch) & 0x0F; 
    return "sm_" + std::to_string(major + minor);
  }

  //==--- Setters ----------------------------------------------------------==//
  
  /// Sets the gpu compute architecture version.
  /// \param[in]  arch  The value of the architecture version, with the format
  ///                   0x{Major}{Minor}.
  VoxxDeviceHost constexpr void setArchitecture(uint64_t arch) noexcept {
    setProperty(arch, Mask::Arch);
  }

  /// Sets the number of streaming multiprocessors for the GPU.
  /// \param[in]  sms   The number of streaming multiprocessors.
  VoxxDeviceHost constexpr void setMultiprocessors(uint64_t sms) noexcept {
    setProperty(sms, Mask::Multiprocessors);
  }

  /// Sets the number of physical cores per multiprocessor.
  /// \param[in]  cores   The number of physical cores per multiprocessor.
  VoxxDeviceHost constexpr void 
  setCoresPerMultiprocessor(uint64_t cores) noexcept {
    setProperty(cores, Mask::CoresPerSm);
  }

  /// Prints the CPU information.
  void print(Io::Format format) const {
    using namespace Voxx::Utility::Io;
    switch (format) {
      case Format::Hexadecimal : printRaw()     ; break;
      case Format::Readable    : printReadable(); break;
      default : break;
    }
  }
 
 private:
  std::size_t Value;

  /// The Mask enum defines the bit masks for the fields.
  enum Mask : uint64_t {
    Arch            = 0x0000000000000FF, //!< Architecture    : Bits [00 - 07].
    Multiprocessors = 0x000000000FFFF00, //!< Multiprocessors : Bits [08 - 15].
    CoresPerSm      = 0x00000FFFF000000  //!< Physical cores  : Bits [16 - 23].
  };

  //==--- Methods ----------------------------------------------------------==//
  
  /// Sets the bits for a property, by first clearing the property and then
  /// setting it.
  /// \param[in] value The value to set the property to.
  /// \param[in] mask  The mask for the property.
  VoxxDeviceHost constexpr void 
  setProperty(uint64_t value, uint64_t mask) noexcept {
    Value = (Value & (~mask)) | ((value << firstSetBitIndex(mask)) & mask);
  }

  /// Gets one of the properties.
  /// \param[in] mask  The mask for the property.
  VoxxDeviceHost constexpr auto 
  getProperty(uint64_t mask) const noexcept -> uint64_t {
    return (Value & mask) >> firstSetBitIndex(mask);
  }

  /// Fills a GpuInfo struct with information for a GPU.
  /// \param[in]  deviceNumber  The device number to get the information for.
  void fillGpuInfo(int deviceNumber) {
#if defined(VoxxCudaSupported)
    int deviceCount = 0;
    VoxxCudaErrorCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceNumber >= deviceCount) {
      printf(
        "\nError: Device: %03i is larger than the %03i Gpus in the system\n",
        deviceNumber, deviceCount);
      return;
    }

    cudaSetDevice(deviceNumber);
    cudaDeviceProp deviceProps;
    VoxxCudaErrorCheck(cudaGetDeviceProperties(&deviceProps, deviceNumber));

    // Rewritten version from cuda samples:
    auto coresForArch = [] (int major, int minor) {
      enum { Fermi = 2, Kepler = 3, Maxwell = 5, Pascal = 6 };
      switch (major) {
        case Fermi   : return minor == 0 ? 32 : 48;
        case Kepler  : return 192;
        case Maxwell : return 128;
        case Pascal  : return minor == 0 ? 64 : 128;
        default      : return -1;
      } 
    };
    setArchitecture((deviceProps.major << 4) + deviceProps.minor);
    setMultiprocessors(deviceProps.multiProcessorCount);
    setCoresPerMultiprocessor(coresForArch(deviceProps.major,
                                           deviceProps.minor));
#endif // VoxxCudaSupported
  }

  //==--- Printing ---------------------------------------------------------==//
  
  /// Prints the information in a raw format.
  void printRaw() const {
    using namespace Voxx::Utility::Io;
    using Out = Output<OutputMode::Terminal>;
    const auto deviceName = std::string("Gpu") + std::to_string(0);
    printf(Out::hexFormat, deviceName.c_str(), Value);
  }
  
  /// Prints the information in a readable format.
  void printReadable() const {
    using namespace Voxx::Utility::Io;
    using Out = Output<OutputMode::Terminal>;

    Out::banner();
    printf(Out::longFormat, "Gpu Number"     , uint64_t{0}                 );
    printf(Out::strFormat , "Architecture"   , architectureString().c_str());
    printf(Out::longFormat, "Processors"     , multiprocessors()           );
    printf(Out::longFormat, "Cores/processor", coresPerMultiProcessor()    );
    printf(Out::longFormat, "Total Cores"    , physicalCores()             );
    Out::banner();   
  }
};

}} // namespace Voxx::Utility