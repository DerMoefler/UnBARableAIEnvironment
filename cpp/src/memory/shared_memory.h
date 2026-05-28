#pragma once
#include "shared_memory_impl.h"

namespace UnBARableAI {

namespace memory {

template<SharedMemoryImpl T>
class SharedMemory {
public:
    /**
     * \brief Creates a shared memory region.
     */
    SharedMemory(void)
        : m_sharedMemoryImpl()
    {}

private:
    T m_sharedMemoryImpl;
};

}; // namespace memory

}; // namespace UnBARableAI
