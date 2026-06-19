#include <gtest/gtest.h>
#include <type_traits>
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

// TEST(SerializationTest, VectorIntegral) {
//     using Vector = std::vector<uint8_t>;

//     Vector data = { 0xBA, 0x52, 0xAB, 0x1E };

//     auto serialized = SerializeInformation<Vector>::serialize(data);
//     std::vector<std::byte> expected = { std::byte(0x00), std::byte(0x00), std::byte(0x00), std::byte(0x00), 
//                                                 std::byte(0x00), std::byte(0x00), std::byte(0x00), std::byte(0x04),
//                                                 std::byte(0xBA), std::byte(0x52), std::byte(0xAB), std::byte(0x1E) };
//     EXPECT_EQ(expected, serialized);

// }

TEST(SerializationTest, VectorGet) {
    using Vector = std::vector<uint8_t>;
    using VectorSI = SerializeInformation<Vector>;

    constexpr auto lengthKey    = std::type_identity<VectorSI::F_Length> {};
    constexpr auto elementsKey  = std::type_identity<VectorSI::F_Elements> {};


    Vector data = { 0xBA, 0x52, 0xAB, 0x1E };

    size_t size = VectorSI::get(lengthKey, data);
    EXPECT_EQ(size, data.size());

    size_t elementSize = VectorSI::getSize(elementsKey, data);
    EXPECT_EQ(elementSize, data.size());
    
    for(int i = 0; i < size; i++) {
        auto element = VectorSI::get(elementsKey, data, i);
        EXPECT_EQ(element, data[i]);
    }
}

TEST(SerializationTest, SimpleVectorSize) {
    using Vector = std::vector<uint8_t>;

    Vector data = { 0xBA, 0x52, 0xAB, 0x1E };

    size_t fieldSize = FieldSizes<Vector>::get(data);
    EXPECT_EQ(12, fieldSize);
}

struct ComplexBase {
    std::vector<std::vector<std::vector<uint8_t>>> tensor3;
};

struct ComplexA : ComplexBase {};

template<>
struct serialization::SerializeInformation<ComplexA> {
    inline static constexpr bool c_serializable = true;

    using Type = ComplexA;

    struct F_Tensor3 : public detail::Field_t {
        using Type = std::vector<std::vector<std::vector<uint8_t>>>;
    };

    using Fields = detail::Fields<F_Tensor3>;
    
    inline static constexpr decltype(auto) get(std::type_identity<F_Tensor3>, const Type& v) noexcept {
        return v.tensor3;
    }
};

TEST(SerializationTest, ComplexA) {
    
    ComplexA data{
        {{
            // Matrix 0: 2 x 4
            {
                {  1,  2,  3,  4 },
                {  5,  6,  7,  8 }
            }
        }}
    };

    size_t size = FieldSizes<ComplexA>::get(data);
    EXPECT_EQ(sizeof(memory::link_t), size);
}

struct ComplexB : ComplexBase {};

template<>
struct serialization::SerializeInformation<ComplexB> {
    inline static constexpr bool c_serializable = true;

    using Type = ComplexB;

    struct F_Tensor3 : public detail::Field_t {
        using Type = std::vector<std::vector<std::vector<uint8_t>>>;
        inline static constexpr bool c_inline = false;
    };

    using Fields = detail::Fields<F_Tensor3>;
    
    inline static constexpr decltype(auto) get(std::type_identity<F_Tensor3>, const Type& v) noexcept {
        return v.tensor3;
    }
};

TEST(SerializationTest, ComplexB) {
    
    ComplexB data{
        {{
            // Matrix 0: 2 x 4
            {
                {  1,  2,  3,  4 },
                {  5,  6,  7,  8 }
            }
        }}
    };

    size_t size = FieldSizes<ComplexB>::get(data);
    EXPECT_EQ(sizeof(memory::link_t), size);
}

} // namespace UnBARableAI
