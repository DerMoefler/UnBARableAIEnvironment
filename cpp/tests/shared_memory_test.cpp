#include <gtest/gtest.h>
#include "memory/shared_memory.h"
#include "memory/shared_memory_unix.h"

namespace UnBARableAI {

using namespace memory;

TEST(SharedMemoryTest, Constructor) {
    SharedMemory<SharedMemoryUnix>();
}

} // namespace UnBARableAI
