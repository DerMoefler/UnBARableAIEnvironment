#include <gtest/gtest.h>
#include "memory/shared_memory.h"
#include "memory/shared_memory_posix.h"

namespace UnBARableAI {

using namespace memory;

TEST(SharedMemoryTest, Constructor) {
    SharedMemory<SharedMemoryPosix>("test");
}

} // namespace UnBARableAI
