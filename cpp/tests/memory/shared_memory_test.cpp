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

    auto funcForEachNode = [&](const auto& node) {
        using Node = std::remove_cvref_t<decltype(node)>;
        if constexpr (Node::isFullyInlined) {
        }
        else {
        }
    };

    if (::testing::Test::HasFailure()) {
        std::ifstream file(std::string("/dev/shm/" + std::string(name)), std::ios::binary);
        if (!file) {
            std::cout << "Cannot open shm?!?";
        }
        else {
            debug::hexdump(std::cout, file);
        }
    }

    SharedMemoryType::remove(std::string(name));
}

TEST_F(SharedMemoryTest, OpenComplexB) {
    using SI = serialization::SerializeInformation<ComplexB>;
    constexpr std::string_view name = "/shm-test-open-complexb";

    SharedMemoryType shm = SharedMemoryType::create(name);
    id::id_t serializableId = shm.write(data);

    SharedMemoryType shmOpen = SharedMemoryType::open(name);

    EXPECT_NO_THROW({
        auto layout = std::get<serialization::Layout<ComplexB>>(shmOpen.getLayout(serializableId));
    });

    if (::testing::Test::HasFailure()) {
        std::ifstream file(std::string("/dev/shm/" + std::string(name)), std::ios::binary);
        if (!file) {
            std::cout << "Cannot open shm?!?";
        }
        else {
            debug::hexdump(std::cout, file);
        }
    }

    SharedMemoryType::remove(std::string(name));
}

}  // namespace UnBARableAINS
