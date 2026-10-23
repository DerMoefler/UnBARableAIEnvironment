#ifndef SHM_HEADER_H_
#define SHM_HEADER_H_

#include <cstdint>

namespace UnBARableAINS {

namespace memory {

/// \brief A sort of tag type only used to de-/serialize
class ShmHeader {
public:
    /// \brief Magic number for the UnBARableAI.
    static constexpr uint32_t c_UnBARableAI_magic = 0xBA52ab1e;

    /// \brief Major version.
    static constexpr uint8_t c_version_major = 0x00;
    /// \brief Minor version.
    static constexpr uint8_t c_version_minor = 0x00;
    /// \brief Patch version.
    static constexpr uint16_t c_version_patch = 0x0001;

    /// \brief Combines \ref c_version_major, \ref c_version_minor and \ref c_version_patch into one
    /// 32 bit version.
    static constexpr uint32_t c_version = static_cast<uint32_t>(c_version_major) << 24 |
                                          static_cast<uint32_t>(c_version_minor) << 16 |
                                          static_cast<uint32_t>(c_version_patch);
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SHM_HEADER_H_
