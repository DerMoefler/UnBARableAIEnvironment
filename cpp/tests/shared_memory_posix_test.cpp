#include <gtest/gtest.h>
#include "memory/shared_memory_posix.h"

namespace UnBARableAINS {

using namespace memory;

TEST(SharedMemoryPosixTest, writeSegment) {
    SharedMemoryPosix shm("/posix-test");
    
    std::vector<std::byte> data {
        std::byte{0xba},
        std::byte{0x52},
        std::byte{0xab},
        std::byte{0x1e},
    };
    shm.writeSegment(data);
    shm.writeSegment(data);
    shm.writeSegment(data);
}

} // namespace UnBARableAI
