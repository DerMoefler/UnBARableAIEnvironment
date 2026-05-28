#pragma once
#include <string>

namespace UnBARableAI {

namespace memory {

/**
 * \brief Uses POSIX shared memory.
 *
 * People seem to prefer POSIX shared memory over system V's, which is used by \ref SharedMemoryUnix.
 */
class SharedMemoryPosix {
public:
    SharedMemoryPosix(std::string_view name);

private:
    /// \brief Name of the shared memory region.
    std::string m_name;
    /// \brief Id returned by shm_open.
    int m_id;
};

}; // namespace memory

}; // namespace UnBARableAI
