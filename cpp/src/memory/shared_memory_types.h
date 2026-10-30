#pragma once
#include <cstdint>
#include <cstddef>
#include <span>

namespace UnBARableAINS {

namespace memory {

/// \brief Type alias for the type used for identifiers in a shared memory region.
using id_t = uint32_t;

/**
 * \brief Type alias for a 'link', which is specifically a relative pointer linking objects
 * together, e.g. the segment table or a segment itself.
 *
 * \note Raw pointers are not usable in shared memory, which is why we use relative offsets to the
 * shared memory's start.
 */
using link_t = id_t;

/**
 * \brief Type alias for a specific position in the shared memory.
 *
 * The position starts at 0 from the shared memory's start. Conceptually very similar to \ref
 * link_t. The difference is, that a \ref link_t is specific to linking binary objects like the
 * segment table or segments themselves together. A position is effectively the general term, while
 * \ref link_t is a semantically special position_t.
 */
using position_t = id_t;

/// \brief Alias for a view to some data (e.g. to write to a segment).
using DataView = const std::span<const std::byte>;

};  // namespace memory

};  // namespace UnBARableAINS
