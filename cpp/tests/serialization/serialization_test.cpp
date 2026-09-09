#include <gtest/gtest.h>
#include <type_traits>
#include "serialization/serialize_information.h"
#include "serialization/layout.h"
#include "serialization/debug/layout_dump.hpp"
#include "complex_type.h"

namespace UnBARableAINS {

namespace serialization {

namespace test {

template <size_t size>
using Serialized = std::array<std::byte, size>;

TEST(SerializationTest, Integrals) {
    constexpr Serialized<1> serialized8 = SerializeInformation<uint8_t>::serialize(0xFF);
    constexpr Serialized<1> expected8 = {std::byte(0xFF)};
    EXPECT_EQ(expected8, serialized8);

    constexpr Serialized<2> serialized16 = SerializeInformation<uint16_t>::serialize(0xA0B1);
    constexpr Serialized<2> expected16 = {std::byte(0xA0), std::byte(0xB1)};
    EXPECT_EQ(expected16, serialized16);
}

// TEST(SerializationTest, VectorIntegral) {
//     using Vector = std::vector<uint8_t>;

//     Vector data = { 0xBA, 0x52, 0xAB, 0x1E };

//     auto serialized = SerializeInformation<Vector>::serialize(data);
//     std::vector<std::byte> expected = { std::byte(0x00), std::byte(0x00), std::byte(0x00),
//     std::byte(0x00),
//                                                 std::byte(0x00), std::byte(0x00),
//                                                 std::byte(0x00), std::byte(0x04),
//                                                 std::byte(0xBA), std::byte(0x52),
//                                                 std::byte(0xAB), std::byte(0x1E) };
//     EXPECT_EQ(expected, serialized);

// }

TEST(SerializationTest, VectorGet) {
    using Vector = std::vector<uint8_t>;
    using VectorSI = SerializeInformation<Vector>;

    constexpr auto lengthKey = std::type_identity<VectorSI::F_Length>{};
    constexpr auto elementsKey = std::type_identity<VectorSI::F_Elements>{};

    Vector data = {0xBA, 0x52, 0xAB, 0x1E};

    size_t size = VectorSI::get(lengthKey, data);
    EXPECT_EQ(size, data.size());

    size_t elementSize = VectorSI::getSize(elementsKey, data);
    EXPECT_EQ(elementSize, data.size());

    for (int i = 0; i < size; i++) {
        auto element = VectorSI::get(elementsKey, data, i);
        EXPECT_EQ(element, data[i]);
    }
}

TEST(SerializationTest, SimpleVectorLayout) {
    using Vector = std::vector<uint8_t>;
    using SI = SerializeInformation<Vector>;

    Vector data = {0xBA, 0x52, 0xAB, 0x1E};

    auto layout = Layout<Vector>{data};
    // Length layout
    auto lengthLayout = layout.get(std::type_identity<SI::F_Length>{});
    EXPECT_EQ(lengthLayout.inlineSize, sizeof(size_t));
    EXPECT_EQ(lengthLayout.deepSize, lengthLayout.inlineSize);
    EXPECT_EQ(lengthLayout.offset, 0);
    // Elements layout
    auto elementsLayout = layout.get(std::type_identity<SI::F_Elements>{});
    EXPECT_EQ(elementsLayout.inlineSize, data.size() * sizeof(Vector::value_type));
    EXPECT_EQ(elementsLayout.deepSize, elementsLayout.inlineSize);
    EXPECT_EQ(elementsLayout.offset, lengthLayout.inlineSize);
    // Overall layout
    EXPECT_EQ(layout.getInlinedSize(), lengthLayout.inlineSize + elementsLayout.inlineSize);
    EXPECT_EQ(layout.getDeepSize(), layout.getInlinedSize());
}

// TEST(SerializationTest, ComplexA) {
//     ComplexA data{{{// Matrix 0: 2 x 4
//                     {{1, 2, 3, 4}, {5, 6, 7, 8}}}}};
//
//     FAIL() << "Not implemented yet";
// }

// TODO REWRITE THIS UGLY ASS TEST
TEST(SerializationTest, ComplexB) {
    using SI = SerializeInformation<ComplexB>;
    using Vector3SI = SerializeInformation<SI::F_Tensor3::Type>;
    using Vector2SI = SerializeInformation<typename Vector3SI::F_Element::Type>;
    using Vector1SI = SerializeInformation<typename Vector2SI::F_Element::Type>;
    ComplexB data{{{// Matrix 0: 2 x 4
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
    // Overall Layout
    Layout<ComplexB> layout{data};
    EXPECT_EQ(layout.getInlinedSize(), 4);
    // Tensor3 Layout
    auto tensor3Layout = layout.get(std::type_identity<SI::F_Tensor3>{});
    EXPECT_EQ(tensor3Layout.inlineSize, 4);
    // Tensor3::Type layout
    auto firstChild = tensor3Layout.child;
    EXPECT_EQ(firstChild->getInlinedSize(),
              sizeof(size_t) + sizeof(memory::link_t) * data.tensor3.size());
    // Tensor3::Type::Length layout
    auto firstLengthLayout = firstChild->get(std::type_identity<Vector3SI::F_Length>{});
    EXPECT_EQ(firstLengthLayout.inlineSize, sizeof(size_t));
    EXPECT_EQ(firstLengthLayout.deepSize, sizeof(size_t));
    EXPECT_EQ(firstLengthLayout.offset, 0);
    static_assert(decltype(firstLengthLayout)::isFieldInlined,
                  "Length of first vector is not inlined!");
    // Tensor3::Type::Elements layout
    auto firstElementsLayout = firstChild->get(std::type_identity<Vector3SI::F_Elements>{});
    EXPECT_EQ(firstElementsLayout.inlineSize, sizeof(memory::link_t) * data.tensor3.size());
    // EXPECT_EQ(firstElementsLayout.deepSize, 8);
    EXPECT_EQ(firstElementsLayout.offset, firstLengthLayout.inlineSize);
    static_assert(decltype(firstElementsLayout)::isFieldInlined,
                  "Elements of first vector is not inlined!");

    for (int i = 0; i < data.tensor3.size(); i++) {
        auto matrixLayout = firstElementsLayout.children[i];

        auto matrixLengthLayout = matrixLayout->get(std::type_identity<Vector2SI::F_Length>{});
        EXPECT_EQ(matrixLengthLayout.inlineSize, sizeof(size_t));
        EXPECT_EQ(matrixLengthLayout.deepSize, sizeof(size_t));
        EXPECT_EQ(matrixLengthLayout.offset, 0);
        auto matrixElementsLayout = matrixLayout->get(std::type_identity<Vector2SI::F_Elements>{});
        EXPECT_EQ(matrixElementsLayout.inlineSize, sizeof(memory::link_t) * data.tensor3[i].size());
        // EXPECT_EQ(matrixElementsLayout.deepSize, 8);
        EXPECT_EQ(matrixElementsLayout.offset, firstLengthLayout.inlineSize);

        for (int j = 0; j < data.tensor3[i].size(); j++) {
            auto rowLayout = matrixElementsLayout.children[j];

            auto rowLengthLayout = rowLayout->get(std::type_identity<Vector1SI::F_Length>{});
            EXPECT_EQ(rowLengthLayout.inlineSize, sizeof(size_t));
            EXPECT_EQ(rowLengthLayout.deepSize, sizeof(size_t));
            EXPECT_EQ(rowLengthLayout.offset, 0);
            auto rowElementsLayout = rowLayout->get(std::type_identity<Vector1SI::F_Elements>{});
            EXPECT_EQ(rowElementsLayout.inlineSize,
                      sizeof(memory::link_t) * data.tensor3[i][j].size());
            // EXPECT_EQ(rowElementsLayout.deepSize, 8);
            EXPECT_EQ(rowElementsLayout.offset, firstLengthLayout.inlineSize);
            // A row is std::vector<int> where the ints should be fully inlined, meaning there are
            // no more layouts.
            EXPECT_TRUE(rowElementsLayout.children.empty());
        }
    }
}

}  // namespace test

}  // namespace serialization

}  // namespace UnBARableAINS
