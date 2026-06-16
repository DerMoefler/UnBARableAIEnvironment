#include <gtest/gtest.h>
#include "serialization/serialize_information.h"

namespace UnBARableAINS {

using namespace serialization;

template<size_t size>
using Serialized = std::array<std::byte, size>;

TEST(SerializationTest, Integrals) {
    constexpr Serialized<1> serialized8 = SerializeInformation<uint8_t>::serialize(0xFF);
    constexpr Serialized<1> expected8 = { std::byte(0xFF) };
    EXPECT_EQ(expected8, serialized8);

    constexpr Serialized<2> serialized16 = SerializeInformation<uint16_t>::serialize(0xA0B1);
    constexpr Serialized<2> expected16 = { std::byte(0xA0), std::byte(0xB1) };
    EXPECT_EQ(expected16, serialized16);
}

TEST(SerializationTest, VectorIntegral) {
    using Vector = std::vector<uint8_t>;

    Vector data = { 0xBA, 0x52, 0xAB, 0x1E };

    auto serialized = SerializeInformation<Vector>::serialize(data);
    std::vector<std::byte> expected = { std::byte(0x00), std::byte(0x00), std::byte(0x00), std::byte(0x00), 
                                                std::byte(0x00), std::byte(0x00), std::byte(0x00), std::byte(0x04),
                                                std::byte(0xBA), std::byte(0x52), std::byte(0xAB), std::byte(0x1E) };
    EXPECT_EQ(expected, serialized);

}

} // namespace UnBARableAI
