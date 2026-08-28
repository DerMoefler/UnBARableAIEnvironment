#include <gtest/gtest.h>
#include "memory/shared_memory.h"
#include "memory/shared_memory_posix.h"
#include "../serialization/complex_type.h"

namespace UnBARableAINS {

using namespace memory;

TEST(SharedMemoryTest, Constructor) {
    SharedMemory<SharedMemoryPosix, serialization::test::ComplexB> shm{"/shm-test"};
    serialization::test::ComplexB data{
        {{
            // Matrix 0: 2 x 4
            {
                {  1,  2,  3,  4 },
                {  5,  6,  7,  8 }
            },
            // Matrix 1: 3 x 3
            {
                {  10,  20,  30},
                {  40,  50,  60},
                {  70,  80,  90},
            },
            // Matrix 2: 4 x 1
            {
                {  100},
                {  200},
                {  300},
                {  400},
            }
        }}
    };
    shm.write(data);
}

} // namespace UnBARableAI
