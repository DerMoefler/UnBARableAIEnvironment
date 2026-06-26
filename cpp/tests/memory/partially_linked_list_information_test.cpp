#include <gtest/gtest.h>
#include "memory/partially_linked_list_information.h"

namespace UnBARableAINS {

using namespace memory;

using PllInformation = PartiallyLinkedListInformation;


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

TEST(PartiallyLinkedListInformationTest, FullHead) {
    constexpr position_t    memoryStart = 0;
    constexpr size_t        headerSize  = 2;
    constexpr size_t        dataSize    = 14;
    PllInformation plli{memoryStart, headerSize, dataSize};
    plli.advanceHead(dataSize);
    EXPECT_TRUE(plli.isListFull());
    
    constexpr position_t    extensionMemoryStart = 30;
    constexpr size_t        extensionHeaderSize  = 5;
    constexpr size_t        extensionDataSize    = 17;
    
    plli.extend(extensionMemoryStart, extensionHeaderSize, extensionDataSize);
    EXPECT_FALSE(plli.isListFull());
    EXPECT_EQ(plli.getHead(), extensionMemoryStart + extensionHeaderSize);
}

} // namespace UnBARableAI
