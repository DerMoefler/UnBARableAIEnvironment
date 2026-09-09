#include <gtest/gtest.h>
#include "memory/shared_memory_posix.h"

namespace UnBARableAINS {

using namespace memory;

using SInformation = SegmentInformation;

TEST(SegmentInformationTest, InvalidConstructor) {
    constexpr size_t id = 2;

    SInformation si{id};
    ASSERT_FALSE(si.isValid());
}

TEST(SegmentInformationTest, SingleSegment) {
    constexpr size_t id = 2;
    constexpr position_t memoryStart = 10;
    constexpr size_t dataSize = 24;

    SInformation si{id, memoryStart, dataSize};
    ASSERT_TRUE(si.isValid());
    ASSERT_EQ(si.getMemoryStart(), memoryStart);
}

TEST(SegmentInformationTest, Initialize) {
    constexpr size_t id = 2;
    constexpr position_t memoryStart = 10;
    constexpr size_t dataSize = 24;

    SInformation si{id};
    ASSERT_FALSE(si.isValid());

    si.initialize(memoryStart, dataSize);

    ASSERT_TRUE(si.isValid());
}

TEST(SegmentInformationTest, ThrowInvalid) {
    constexpr std::string_view errorMessage =
        "Object has to be valid for the request operation, but is invalid.";
    constexpr size_t id = 2;

    SInformation si{id};
    ASSERT_FALSE(si.isValid());

    try {
        si.getMemoryStart();
        FAIL();
    } catch (const std::logic_error& e) {
        EXPECT_STREQ(e.what(), errorMessage.data());
    }
}

TEST(SegmentInformationTest, ThrowValid) {
    constexpr std::string_view errorMessage =
        "Object has to be invalid for the request operation, but is valid.";

    constexpr size_t id = 2;
    constexpr position_t memoryStart = 10;
    constexpr size_t dataSize = 24;

    SInformation si{id, memoryStart, dataSize};
    ASSERT_TRUE(si.isValid());

    try {
        si.initialize(memoryStart, dataSize);
        FAIL();
    } catch (const std::logic_error& e) {
        EXPECT_STREQ(e.what(), errorMessage.data());
    }
}

TEST(SegmentInformationTest, SequentialReadSafety) {
    constexpr size_t id = 0;
    constexpr position_t memoryStart = 0;
    constexpr size_t dataSize = 8;

    SegmentInformation si{id, memoryStart, dataSize};

    EXPECT_TRUE(si.isSequentialReadSafe(0, 4));
    EXPECT_TRUE(si.isSequentialReadSafe(0, 8));
    EXPECT_TRUE(si.isSequentialReadSafe(4, 4));
    EXPECT_FALSE(si.isSequentialReadSafe(4, 5));
    EXPECT_FALSE(si.isSequentialReadSafe(9, 1));
}

}  // namespace UnBARableAINS
