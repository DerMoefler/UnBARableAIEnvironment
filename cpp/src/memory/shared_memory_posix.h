#pragma once
#include <string>
#include <cstdint>

namespace UnBARableAI {

namespace memory {

/**
 * \brief Uses POSIX shared memory.
 *
 * People seem to prefer POSIX shared memory over system V's, which is used by \ref SharedMemoryUnix.
 */
class SharedMemoryPosix {
public:
    /**
     * \brief Acquire shared memory using shm_open().
     * \param name The name of the shared memory region.
     * 
     * The \p name must be unique (i.e. cannot exist already), otherwise an error occurs.
     * \throws std::system_error on shmp_open fail.
     */
    SharedMemoryPosix(std::string_view name);

    /**
     * \brief Unlink POSIX memory using shm_unlink().
     *
     * shm_unlink is simply called and eventual failures are ignored.
     */
    ~SharedMemoryPosix(void);

    void writeSegment(std::span<const std::byte> data);

private:
    /// \brief Constant by which the size is increased when needed.
    inline static constexpr int64_t     c_size_increase = 4096;

    /// \brief Name of the shared memory region.
    std::string                         m_name;

    /// \brief Id returned by shm_open.
    int                                 m_id;

    /// \brief Size of the shared memory region.
    uint64_t                            m_size;
};

}; // namespace memory

}; // namespace UnBARableAI
