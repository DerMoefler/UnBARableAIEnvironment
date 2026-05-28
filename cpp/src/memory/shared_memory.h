#pragma once
#include <string_view>

#include "shared_memory_impl.h"

namespace UnBARableAI {

namespace memory {

template<SharedMemoryImpl T>
class SharedMemory {
public:
    /**
     * \brief Creates a shared memory region.
     */
    SharedMemory(std::string_view name)
        : m_sharedMemoryImpl(name)
    {}

private:
    T m_sharedMemoryImpl;
};

}; // namespace memory

}; // namespace UnBARableAI
