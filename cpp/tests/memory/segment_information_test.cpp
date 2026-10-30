#include <gtest/gtest.h>

#include "memory/segment_information.hpp"

namespace UnBARableAINS {

using namespace memory;

TEST(SegmentInformationTest, InvalidConstructor) {
    constexpr size_t id = 2;

    constexpr SegmentInformation si{id};
    ASSERT_FALSE(si.isValid());
}

TEST(SegmentInformationTest, SingleSegment) {
    constexpr size_t id = 2;
    constexpr position_t memoryStart = 10;
    constexpr size_t dataSize = 24;

    constexpr SegmentInformation si{id, memoryStart, dataSize};
    ASSERT_TRUE(si.isValid());
    ASSERT_EQ(si.getMemoryStart(), memoryStart);
}

TEST(SegmentInformationTest, Initialize) {
    constexpr size_t id = 2;
    constexpr position_t memoryStart = 10;
    constexpr size_t dataSize = 24;

    SegmentInformation si{id};
    ASSERT_FALSE(si.isValid());

    si = si.initialize(memoryStart, dataSize);

    ASSERT_TRUE(si.isValid());
}

}  // namespace UnBARableAINS
