#ifndef UNSIGNED_OF_SIZE_H_
#define UNSIGNED_OF_SIZE_H_

#include <cstddef>
#include <cstdint>

namespace util {

template <std::size_t size>
struct UnsignedOfSize_MF;

template <>
struct UnsignedOfSize_MF<1> {
    using Type = std::uint8_t;
};

template <>
struct UnsignedOfSize_MF<2> {
    using Type = std::uint16_t;
};

template <>
struct UnsignedOfSize_MF<4> {
    using Type = std::uint32_t;
};

template <>
struct UnsignedOfSize_MF<8> {
    using Type = std::uint64_t;
};

template <std::size_t size>
using UnsignedOfSize_T = typename UnsignedOfSize_MF<size>::Type;

}  // namespace util

#endif  // UNSIGNED_OF_SIZE_H_
