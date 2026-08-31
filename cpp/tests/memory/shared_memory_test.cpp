#include "../serialization/complex_type.h"
#include "memory/shared_memory.h"

#include <gtest/gtest.h>

#include "memory/shared_memory_posix.h"

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

TEST_F(SharedMemoryTest, Write) {
    using SI = serialization::SerializeInformation<ComplexB>;
    SharedMemoryType shm = SharedMemoryType::create("/shm-test");
    id::id_t serializableId = shm.write(data);

    serialization::Layout<ComplexB>& mainLayout =
        std::get<serialization::Layout<ComplexB>>(shm.getLayout(serializableId));

    auto mainSegmentId = mainLayout.getSegmentId();
    ASSERT_TRUE(mainSegmentId.has_value());
}

}  // namespace UnBARableAINS
