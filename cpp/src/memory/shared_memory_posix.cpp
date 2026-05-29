#include "shared_memory_posix.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <system_error>

namespace UnBARableAI {

namespace memory {

SharedMemoryPosix::SharedMemoryPosix(std::string_view name) 
    : m_name(name)
    , m_size(0)
{
    m_id = shm_open(m_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
    if(m_id == -1) {
        throw std::system_error(errno, std::generic_category(), "shm_open failed");
    }
}

SharedMemoryPosix::~SharedMemoryPosix(void) {
    shm_unlink(m_name.c_str());
}

}; // namespace memory

}; // namespace UnBARableAI
