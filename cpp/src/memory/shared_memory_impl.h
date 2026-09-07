#pragma once
#include <cstddef>
#include <concepts>
#include <span>
#include <string_view>

#include "../memory/shared_memory_types.h"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Requirements for a shared memory backend.
 *
 * A shared memory implementation is used to write identifiable segments into memory.
 * An implementation should at least provide a version (for its own ABI) at the start
 * of the shared memory.
 */
template <class T>
concept SharedMemoryImpl =
    requires(std::string_view name, T t, std::span<const std::byte> data, id_t id, size_t size) {
        { T::create(name) } -> std::same_as<T>;
        { T::open(name) } -> std::same_as<T>;
        { T::remove(name) } -> std::same_as<void>;
        { t.writeSegment(data) } -> std::convertible_to<id_t>;
        { t.createSegment(size) } -> std::convertible_to<id_t>;
        // { t.deleteSegment(id) }     -> std::same_as<void>;
    };

};  // namespace memory

};  // namespace UnBARableAINS
