#include <gtest/gtest.h>
#include "memory/shared_memory_posix.h"

namespace UnBARableAINS {

using namespace memory;

using PllInformation = SharedMemoryPosix::PartiallyLinkedListInformation;
using position_t = SharedMemoryPosix::position_t;


TEST(PartiallyLinkedListInformationTest, SingleSegment) {
    constexpr position_t    memoryStart = 10;
    constexpr size_t        headerSize  = 2;
    constexpr size_t        dataSize    = 14;

    PllInformation plli{memoryStart, headerSize, dataSize};
    
    
    ASSERT_EQ(plli.getSize(), headerSize + dataSize + PllInformation::c_link_size);
    ASSERT_EQ(plli.getCapacity(), dataSize);
    ASSERT_EQ(plli.getHeaderSize(0), headerSize);

    ASSERT_EQ(plli.getHeaderStart(0), memoryStart);
    ASSERT_EQ(plli.getDataStart(0), memoryStart + headerSize);
    ASSERT_EQ(plli.getDataPosition(0), memoryStart + headerSize);
    ASSERT_EQ(plli.getDataPosition(8), memoryStart + headerSize + 8);
    ASSERT_EQ(plli.getDataPosition(dataSize - 1), memoryStart + headerSize + dataSize - 1);
}

TEST(PartiallyLinkedListInformationTest, getIndexPosition) {
    constexpr position_t    memoryStartA = 10;
    constexpr position_t    memoryStartB = 50;
    constexpr position_t    memoryStartC = 100;
    constexpr size_t        headerSize  = 10;
    constexpr size_t        dataSize    = 16;

    PllInformation plli{memoryStartA, headerSize, dataSize};
    plli.extend(memoryStartB, headerSize, dataSize);
    plli.extend(memoryStartC, headerSize, dataSize);

    EXPECT_EQ(plli.getIndexPosition(memoryStartC + 12), 2 * dataSize + 2);
}

} // namespace UnBARableAI
