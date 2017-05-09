//==--- Voxel/Heterogeneous/Utils.hpp ---------------------- -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  Utils.hpp
/// \brief This file defines utilities for heterogeneous functionality.
//
//==------------------------------------------------------------------------==//

#pragma once

#include "HeteroException.hpp"
#include <Voxel/Utility/Debug.hpp>
#include <Voxel/Utility/Portability.hpp>

namespace Voxx   {
namespace Hetero {

// Host size gpu utility functions.
#if defined(VoxxCudaSupported)

/// Copies \p bytes of data from \p hostPointer to \p devPointer.
/// \param[in]  hostPointer   The host pointer to copy from.
/// \param[in]  devPointer    The device pointer to copy to.
/// \param[in]  bytes         The number of bytes to copy.
/// \tparam     HostPointer   The type of the host pointer.
/// \tparam     DevPointer    The type of the device pointer.
template <typename HostPointer, typename DevPointer>
inline void copyToDevice(const HostPointer* hostPointer,
                         DevPointer*        devPointer ,
                         std::size_t        bytes      ) {
  Debug::catcher([] (const auto* hPointer, const auto* dPointer) {
    if (hPointer == nullptr || dPointer == nullptr)
      throw HeteroException{HeteroException::Type::InvalidCopy};
  }, hostPointer, devPointer);

  VoxxCudaErrorCheck(
    cudaMemcpy(devPointer, hostPointer, bytes, cudaMemcpyHostToDevice));
}

/// Copies \p bytes of data from \p devPointer to \p hostPointer.
/// \param[in]  devPointer    The device pointer to copy from.
/// \param[in]  hostPointer   The host pointer to copy to.
/// \param[in]  bytes         The number of bytes to copy.
/// \tparam     DevPointer    The type of the device pointer.
/// \tparam     HostPointer   The type of the host pointer.
template <typename DevPointer, typename HostPointer>
inline void copyToHost(const DevPointer* devPointer ,
                       HostPointer*      hostPointer,
                       std::size_t       bytes      ) {
  Debug::catcher([] (const auto* hPointer, const auto* dPointer) {
    if (hPointer == nullptr || dPointer == nullptr)
      throw HeteroException{HeteroException::Type::InvalidCopy};
  }, hostPointer, devPointer);

  VoxxCudaErrorCheck(
    cudaMemcpy(hostPointer, devPointer, bytes, cudaMemcpyDeviceToHost));
}

/// This namespace container device related heterogenoeous functionality.
namespace Device {

/// Allocates \p bytes of memory on the device pointer to by \p devPointer.
/// \param[in] devPointer   The device pointer to allocate memory for.
/// \param[in] bytes        The number of bytes to allocate.
/// \tparam    Pointer      The type of the pointer.
template <typename Pointer>
inline void allocate(Pointer** devPointer, std::size_t bytes) {
  // Throw reallocation error in debug mode so we can catch cases where
  // we are allocating a device pointer which points to valid data:
  Debug::catcher([] (const auto* pointer) {
    if (pointer != nullptr)
      throw HeteroException{HeteroException::Type::Realloc};
  }, *devPointer);

  VoxxCudaErrorCheck(cudaMalloc((void**)devPointer, bytes));
}

/// Frees the data pointed to by \p pointer.
/// \param[in] pointer  A pointer to the data to free.
/// \tparam    Pointer  The type of the pointer to free.
template <typename Pointer>
inline void free(Pointer** pointer) noexcept {
  VoxxCudaErrorCheck(cudaFree(*pointer));
}

} // namespace Device

#endif // VoxxCudaSupported

}} // namespace Voxx::Hetero