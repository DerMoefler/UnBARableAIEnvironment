#include <gtest/gtest.h>
#include "memory/shared_memory_posix.h"
#include "memory/shared_memory_types.h"
#include "serialization/serialize_information.h"

namespace UnBARableAINS {

using namespace memory;

class SharedMemoryPosixTest : public ::testing::Test {
protected:
    std::vector<std::byte> data = {std::byte{0xba}, std::byte{0x52}, std::byte{0xab},
                                   std::byte{0x1e}};
};

TEST_F(SharedMemoryPosixTest, Wrtie) {
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

TEST_F(SharedMemoryPosixTest, LinkedSegment) {
    SharedMemoryPosix shm = SharedMemoryPosix::create("/shm-posix-test-linked-segment");

    id_t dataSegmentId = shm.writeSegment(data);
    auto serializedDataSegmentId =
        serialization::SerializeInformation<position_t>::serialize(dataSegmentId);
    id_t linkSegmentId = shm.writeSegment(serializedDataSegmentId);
    EXPECT_EQ(shm.getLinkedSegment(linkSegmentId, 0), dataSegmentId);

    SharedMemoryPosix::remove("/shm-posix-test-linked-segment");
}

TEST_F(SharedMemoryPosixTest, OpenEmpty) {
    SharedMemoryPosix shm = SharedMemoryPosix::create("/shm-posix-test-open-empty");

    SharedMemoryPosix shmOpen = SharedMemoryPosix::open("/shm-posix-test-open-empty");
    EXPECT_EQ(shm.getSegmentsCount(), shmOpen.getSegmentsCount());

    SharedMemoryPosix::remove("/shm-posix-test-open-empty");
}

TEST_F(SharedMemoryPosixTest, OpenRead) {
    SharedMemoryPosix shm = SharedMemoryPosix::create("/shm-posix-test-open-read");

    id_t id = shm.writeSegment(data);
    SharedMemoryPosix shmOpen = SharedMemoryPosix::open("/shm-posix-test-open-read");

    // Explicit test
    auto siCreate = shm.getSegmentInformation(id);
    auto siOpen = shmOpen.getSegmentInformation(id);
    EXPECT_EQ(siCreate.getId(), siOpen.getId());
    EXPECT_EQ(siCreate.isValid(), siOpen.isValid());
    EXPECT_EQ(siCreate.headPosition(), siOpen.headPosition());
    EXPECT_EQ(siCreate.getMemoryStart(), siOpen.getMemoryStart());
    EXPECT_EQ(siCreate.totalSize(), siOpen.totalSize());

    // Test using default operator==
    EXPECT_EQ(shm.getSegmentInformation(id), shmOpen.getSegmentInformation(id));

    SharedMemoryPosix::remove("/shm-posix-test-open-read");
}

TEST_F(SharedMemoryPosixTest, Delete) {
    constexpr std::size_t numSegments = 10;
    constexpr std::size_t segmentSize = 32;
    std::vector<std::byte> sequentialData;
    sequentialData.reserve(numSegments * segmentSize);
    for (std::size_t i = 0; i < numSegments * segmentSize; i++) {
        sequentialData.push_back(static_cast<std::byte>(i));
    }

    std::string shmName = "/shm-posix-test-delete";

    SharedMemoryPosix writer = SharedMemoryPosix::create(shmName);
    SharedMemoryPosix reader = SharedMemoryPosix::open(shmName);

    auto view = std::span{sequentialData};
    for (std::size_t i = 0; i < numSegments; i++) {
        auto segmentView = view.subspan(i * segmentSize, segmentSize);
        writer.writeSegment(segmentView);
    }

    reader.update();
    EXPECT_EQ(reader.getSegmentsCount(), numSegments);

    writer.deleteSegment(2);
    writer.deleteSegment(5);

    reader.update();
    EXPECT_EQ(reader.getSegmentsCount(), numSegments - 2);

    writer.deleteSegment(numSegments);

    reader.update();
    EXPECT_EQ(reader.getSegmentsCount(), numSegments - 3);

    SharedMemoryPosix::remove("/shm-posix-test-delete");
}

}  // namespace UnBARableAINS
>>>>>>> c72de93 (WIP: Changed SegmentInformation.)
