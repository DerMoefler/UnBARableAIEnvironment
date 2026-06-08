#pragma once
#include <cstddef>
#include <concepts>
#include <cstdint>
#include <span>

namespace UnBARableAI {

namespace memory {

/// \todo Move into types.hpp or something

/// \brief Type alias for the type used for identifiers in a shared memory region.
using id_t = uint32_t;

/// \brief Magic number for the UnBARableAI.
inline constexpr uint32_t   c_UnBARableAI_magic = 0xBA52ab1e;

/// \brief Major version.
inline constexpr uint8_t    c_version_major     = 0x00;
/// \brief Minor version.
inline constexpr uint8_t    c_version_minor     = 0x00;
/// \brief Patch version.
inline constexpr uint16_t   c_version_patch     = 0x0001;

/// \brief Combines \ref c_version_major, \ref c_version_minor and \ref c_version_patch into one 32 bit version.
inline constexpr uint32_t   c_version =   static_cast<uint32_t>(c_version_major) << 24
                                        | static_cast<uint32_t>(c_version_minor) << 16
                                        | static_cast<uint32_t>(c_version_patch);

/**
 * \brief Requirements for a shared memory backend.
 *
 * A shared memory implementation is used to write identifiable segments into memory.
 * An implementation should at least provide a version (for its own ABI) at the start
 * of the shared memory.
 */
template<class T>
concept SharedMemoryImpl = requires (T t, std::span<const std::byte> data, id_t id) {
    { t.writeSegment(data) }    -> std::same_as<id_t>;
    // { t.deleteSegment(id) }     -> std::same_as<void>;
};

}; // namespace memory

}; // namespace UnBARableAI
