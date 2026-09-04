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
    SharedMemoryPosix shm = SharedMemoryPosix::create("/posix-test");

    shm.writeSegment(data);
    shm.writeSegment(data);
    shm.writeSegment(data);
    shm.createSegment(0xFF);
}

TEST_F(SharedMemoryPosixTest, Read) {
    SharedMemoryPosix shm = SharedMemoryPosix::create("/posix-read-test");
    std::cout << "Create successfull\n";
    id_t id = shm.writeSegment(data);
    std::cout << "Write successfull\n";
    auto result = shm.readSegment(id);
    EXPECT_EQ(data, result);
}

TEST_F(SharedMemoryPosixTest, Open) {
    SharedMemoryPosix shm = SharedMemoryPosix::create("/posix-open-test");
    SharedMemoryPosix shmOpen = SharedMemoryPosix::open("/posix-open-test");
}

}  // namespace UnBARableAINS
