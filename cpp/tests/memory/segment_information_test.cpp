#include <gtest/gtest.h>
#include "memory/shared_memory_posix.h"

namespace UnBARableAINS {

using namespace memory;

using SInformation = SharedMemoryPosix::SegmentInformation;

TEST(SegmentInformationTest, InvalidConstructor) {
    constexpr size_t        id  = 2;

    SInformation si{id};
    ASSERT_FALSE(si.isValid());
}

TEST(SegmentInformationTest, SingleSegment) {
    constexpr size_t        id  = 2;
    constexpr position_t    memoryStart = 10;
    constexpr size_t        dataSize    = 24;

    SInformation si{id, memoryStart, dataSize};
    ASSERT_TRUE(si.isValid());
    ASSERT_EQ(si.getMemoryStart(), memoryStart);
}

TEST(SegmentInformationTest, Initialize) {
    constexpr size_t        id  = 2;
    constexpr position_t    memoryStart = 10;
    constexpr size_t        dataSize    = 24;

    SInformation si{id};
    ASSERT_FALSE(si.isValid());

    si.initialize(memoryStart, dataSize);

    ASSERT_TRUE(si.isValid());
}

TEST(SegmentInformationTest, ThrowInvalid) {
    constexpr std::string_view errorMessage = "Object has to be valid for the request operation, but is invalid.";
    constexpr size_t        id  = 2;

    SInformation si{id};
    ASSERT_FALSE(si.isValid());

    try {
        si.getMemoryStart();
        FAIL();
    }
    catch(const std::logic_error& e) {
        EXPECT_STREQ(e.what(), errorMessage.data());
    }  
}

TEST(SegmentInformationTest, ThrowValid) {
    constexpr std::string_view errorMessage = "Object has to be invalid for the request operation, but is valid.";

    constexpr size_t        id  = 2;
    constexpr position_t    memoryStart = 10;
    constexpr size_t        dataSize    = 24;

    SInformation si{id, memoryStart, dataSize};
    ASSERT_TRUE(si.isValid());

    try {
        si.initialize(memoryStart, dataSize);
        FAIL();
    }
    catch(const std::logic_error& e) {
        EXPECT_STREQ(e.what(), errorMessage.data());
    }  
}

} // namespace UnBARableAI
