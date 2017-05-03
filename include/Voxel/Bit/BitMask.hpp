//==--- Voxel/Bit/BitMask.hpp ------------------------------ -*- C++ -*- ---==//
//            
//                                    Voxel
//
//                        Copyright (c) 2017 Rob Clucas
//  
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  BitMask.hpp
/// \brief This file defines a dynamically sizable bitmask class.
//
//==------------------------------------------------------------------------==//

#pragma once

#include "BitManip.hpp"
#include <vector>

namespace Voxx {

/// The BitMask class defines a dynamically resizable, bitmask class which can
/// scale to arbitrary size. Internally, the bitmask is stored contiguously,
/// i.e the first byte is a an 8 wide mask, where the first bit in the byte is
/// flag 0, and the last bit is flag 7. If the required number of bits is a
/// multiple of 8, then the required number of bytes for the class is exactly
/// the number of bits in the mask divided by 8, otherwise there will be 
/// ``totalBits % 8`` bits of padding.
class BitMask {
 private:
  struct Manipulator;
 public:
  /// Defines the type of the buckets.
  using BucketType                               = uint8_t;
  /// Defines the number of elements per bucket.
  static constexpr std::size_t elementsPerBucket = sizeof(BucketType) * 8;

  /// Constructor -- sets the initial size of the mask.
  /// \param[in]  initialSize   The initial size of the mask.
  /// \param[in]  set           If the mask values are initially set or not.
  explicit BitMask(std::size_t initialSize, bool set = false) 
  : Data(initialSize / elementsPerBucket +
         std::min(initialSize % elementsPerBucket, 1ul), set),
    Elements(initialSize) {}
  
  /// Returns the number of bits in the mask.
  std::size_t size() const noexcept { return Elements; }

  /// Returns the number of bits which can be stored in the mask without
  /// reallocating.
  std::size_t capacity() const noexcept {
    return Data.size() * elementsPerBucket;
  };
  
  /// Grows the mask by \p amount bits, resizing if necessary.
  /// \param[in] amount   The number of bits to grow by.
  void grow(std::size_t amount) {
      auto diff = amount - (capacity() - size());
      size_t newBuckets = diff / elementsPerBucket
                        + std::min(diff % elementsPerBucket, 1ul);
      for (size_t i = 0; i < newBuckets; ++i)
        Data.emplace_back(0);
      Elements += amount;
  }

  /// Returns the manipulater which does the appropriate manipulation.
  /// \param[in] i The index of the bit to manipulate.
  Manipulator operator[](uint8_t i) noexcept {
    return Manipulator{Data[i / elementsPerBucket],
                       static_cast<uint8_t>(i % elementsPerBucket)};
  }
  
  /// Prints the bitmask. This always prints the least significant bit (bit 0)
  /// as the rightmost bit, and the most significant as the leftmost.
  void print() {
    printf("0b");
    for (size_t bucketIdx = Data.size(); bucketIdx > 0; --bucketIdx) {
      for (size_t bit = elementsPerBucket; bit > 0; --bit) {
        bool value = getBit(Data[bucketIdx - 1], bit - 1);
        printf("%u", value);
      }
    }
  }

 private:
  std::vector<uint8_t> Data;      //!< The data holding the mask.
  std::size_t          Elements;  //!< The number of elements in the mask.

  /// This is a wrapper class to enable both getting and setting of bits in the
  /// BitMask through the acess operator[].
  struct Manipulator {
    /// Constructor -- sets the bucket to manipulate and the index of the
    /// element in the bucket to manipulate.
    /// \param[in] bucket   A reference to the bucket to manipulate.
    /// \param[in] index    The index of the bit in the bucket to manipulate.
    Manipulator(BucketType& bucket, uint8_t index) noexcept
    : Bucket(bucket), Index(index) {}
    
    /// Overload of the assignment operator to allow the bit at Index in Bucket
    /// to be set to \p value.  
    void operator=(uint8_t value) noexcept {
      setBit(Bucket, Index, value);
    }

    /// Overload of operator bool to allow the bit and Index in Bucket to be
    /// accessed as a flag.
    operator bool() const noexcept {
      return getBit(Bucket, Index);
    }

    BucketType& Bucket;   //!< The bucket to manipulate.
    uint8_t     Index;    //!< The index of the bucket in the bit to manipulate.
  };
};

} // namespace Voxx