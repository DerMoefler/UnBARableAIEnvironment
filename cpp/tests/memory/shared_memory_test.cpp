#include "../serialization/complex_type.h"
#include "memory/shared_memory.h"

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "memory/shared_memory_posix.h"
#include "memory/debug/hexdump.hpp"
#include "serialization/debug/layout_dump.hpp"
#include "serialization/serialize_information.h"
#include "serialization/enum.tpp"

namespace UnBARableAINS {

using namespace memory;

class SharedMemoryTest : public ::testing::Test {
protected:
    using ComplexB = serialization::test::ComplexB;
    using SharedMemoryType = SharedMemory<SharedMemoryPosix, ComplexB>;
    serialization::test::ComplexB data{{{// Matrix 0: 2 x 4
                                         {{1, 2, 3, 4}, {5, 6, 7, 8}},
                                         // Matrix 1: 3 x 3
                                         {
                                             {10, 20, 30},
                                             {40, 50, 60},
                                             {70, 80, 90},
                                         },
                                         // Matrix 2: 4 x 1
                                         {
                                             {100},
                                             {200},
                                             {300},
                                             {400},
                                         }}}};

    using TestAction = serialization::test::TestAction;
    enum class TestEnum : std::uint8_t { ValueA = 0, ValueB = 255 };
};

TEST_F(SharedMemoryTest, WriteComplexB) {
    using SI = serialization::SerializeInformation<ComplexB>;
    constexpr std::string_view name = "/shm-test-write";

    SharedMemoryType shm = SharedMemoryType::create(name);
    id::id_t serializableId = shm.write(data);

    SharedMemoryPosix shmPosix = SharedMemoryPosix::open(name);

    serialization::Layout<ComplexB>& mainLayout =
        std::get<serialization::Layout<ComplexB>>(shm.getLayout(serializableId));

    auto mainSegmentId = mainLayout.getSegmentId();
    ASSERT_TRUE(mainSegmentId.has_value());

    std::optional<SegmentInformation> linkedSegment;

    if (::testing::Test::HasFailure()) {
        std::ifstream file(std::string("/dev/shm/" + std::string(name)), std::ios::binary);
        if (!file) {
            std::cout << "Cannot open shm?!?";
        }
        else {
            memory::debug::hexdump(std::cout, file);
        }
    }

    SharedMemoryType::remove(std::string(name));
}

TEST_F(SharedMemoryTest, OpenComplexB) {
    using SI = serialization::SerializeInformation<ComplexB>;
    constexpr std::string_view name = "/shm-test-open-complexb";

    SharedMemoryType shm = SharedMemoryType::create(name);
    id::id_t serializableId = shm.write(data);

    EXPECT_NO_THROW({
        SharedMemoryType shmOpen = SharedMemoryType::open(name);

        auto writtenLayout =
            std::get<serialization::Layout<ComplexB>>(shm.getLayout(serializableId));
        auto openedLayout =
            std::get<serialization::Layout<ComplexB>>(shmOpen.getLayout(serializableId));

        EXPECT_EQ(writtenLayout, openedLayout);
        serialization::debug::dumpLayout(std::cout, writtenLayout, 0);
        serialization::debug::dumpLayout(std::cout, openedLayout, 0);
    });

    if (::testing::Test::HasFailure()) {
        std::ifstream file(std::string("/dev/shm/" + std::string(name)), std::ios::binary);
        if (!file) {
            std::cout << "Cannot open shm?!?";
        }
        else {
            memory::debug::hexdump(std::cout, file);
        }
    }

    SharedMemoryType::remove(std::string(name));
}

TEST_F(SharedMemoryTest, ReadComplexB) {
    using SI = serialization::SerializeInformation<ComplexB>;
    constexpr std::string_view name = "/shm-test-open-complexb";

    SharedMemoryType shm = SharedMemoryType::create(name);
    id::id_t serializableId = shm.write(data);

    SharedMemoryType shmOpen = SharedMemoryType::open(name);

    auto result = shmOpen.read(serializableId);

    EXPECT_EQ(std::get<ComplexB>(result).tensor3, data.tensor3);

    if (::testing::Test::HasFailure()) {
        std::ifstream file(std::string("/dev/shm/" + std::string(name)), std::ios::binary);
        if (!file) {
            std::cout << "Cannot open shm?!?";
        }
        else {
            memory::debug::hexdump(std::cout, file);
        }
    }

    SharedMemoryType::remove(std::string(name));
}

TEST_F(SharedMemoryTest, Integrals) {
    using Shm = SharedMemory<SharedMemoryPosix, std::size_t, TestAction, TestEnum>;

    constexpr std::string_view name = "/shm-test-integrals";

    Shm shm = Shm::create(name);
    id::id_t testActionId = shm.write(TestAction::ValueC);
    id::id_t testEnumId = shm.write(TestEnum::ValueB);

    std::ifstream file(std::string("/dev/shm/" + std::string(name)), std::ios::binary);
    if (!file) {
        std::cout << "Cannot open shm?!?";
    }
    else {
        memory::debug::hexdump(std::cout, file);
    }

    auto testAction = shm.read(testActionId);
    auto testEnum = shm.read(testEnumId);

    EXPECT_EQ(std::get<TestAction>(testAction), TestAction::ValueC);
    EXPECT_EQ(std::get<TestEnum>(testEnum), TestEnum::ValueB);

    if (::testing::Test::HasFailure()) {
        std::ifstream file(std::string("/dev/shm/" + std::string(name)), std::ios::binary);
        if (!file) {
            std::cout << "Cannot open shm?!?";
        }
        else {
            memory::debug::hexdump(std::cout, file);
        }
    }
    Shm::remove(std::string(name));
}

}  // namespace UnBARableAINS
