#include <gtest/gtest.h>
#include "memory/shared_memory_posix.h"

namespace UnBARableAINS {

using namespace memory;

class SharedMemoryPosixTest : public ::testing::Test {
protected:
    std::vector<std::byte> data = {std::byte{0xba}, std::byte{0x52}, std::byte{0xab},
                                   std::byte{0x1e}};
};

TEST_F(SharedMemoryPosixTest, writeSegment) {
    SharedMemoryPosix shm = SharedMemoryPosix::create("/shm-posix-test-write");

    shm.writeSegment(data);
    shm.writeSegment(data);
    shm.writeSegment(data);
    shm.createSegment(0xFF);

    SharedMemoryPosix::remove("/shm-posix-test-write");
}

TEST_F(SharedMemoryPosixTest, Read) {
    SharedMemoryPosix shm = SharedMemoryPosix::create("/shm-posix-test-read");
    id_t id = shm.writeSegment(data);
    auto result = shm.readSegment(id);
    EXPECT_EQ(data, result);
    SharedMemoryPosix::remove("/shm-posix-test-read");
}

TEST_F(SharedMemoryPosixTest, Open) {
    SharedMemoryPosix shm = SharedMemoryPosix::create("/shm-posix-test-open");
    SharedMemoryPosix shmOpen = SharedMemoryPosix::open("/shm-posix-test-open");
    SharedMemoryPosix::remove("/shm-posix-test-open");
}

}  // namespace UnBARableAINS
