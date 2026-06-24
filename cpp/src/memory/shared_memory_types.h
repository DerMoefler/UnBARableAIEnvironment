#pragma once
#include <cstdint>

namespace UnBARableAINS {

namespace memory {

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
* \brief Type alias for a 'link', which is specifically a relative pointer linking objects together, e.g. the segment table or a segment itself.
*
* \note Raw pointers are not usable in shared memory, which is why we use relative offsets to the shared memory's start.
*/ 
using link_t = id_t;

/** 
* \brief Type alias for a specific position in the shared memory.
*
* The position starts at 0 from the shared memory's start. Conceptually very similar to \ref link_t. The difference is, that
* a \ref link_t is specific to linking binary objects like the segment table or segments themselves together. A position is effectively
* the general term, while \ref link_t is a semantically special position_t.
*/ 
using position_t = id_t;

}; // namespace memory

}; // namespace UnBARableAI
