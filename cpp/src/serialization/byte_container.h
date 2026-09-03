#ifndef BYTE_CONTAINER_H_
#define BYTE_CONTAINER_H_

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace UnBARableAINS {

namespace serialization {

namespace detail {

/**
 * \brief Metafunction to checker, whether a type \p T qualifies as an array of bytes.
 * \tparam T Type to check.
 */
template <typename T>
struct IsByteArray_MF : std::false_type {};

/**
 * \brief Partial specialization for std::array<std::byte>.
 * \tparam N Size of the std::array.
 */
template <size_t N>
struct IsByteArray_MF<std::array<std::byte, N>> : std::true_type {};

}  // namespace detail

/**
 * \brief Concept for a byte container.
 * \tparam T Type to check.
 *
 * Checks whether the type \p T is eligible as a container storing bytes. Currently allows
 * std::array<std::byte, N> for any N and std::vector<std::byte>.
 */
template <typename T>
concept ByteContainer = detail::IsByteArray_MF<std::remove_cvref_t<T>>::value ||
                        std::same_as<std::remove_cvref_t<T>, std::vector<std::byte>>;

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // BYTE_CONTAINER_H_
