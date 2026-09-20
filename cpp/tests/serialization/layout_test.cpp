#include <gtest/gtest.h>
#include <type_traits>

#include "serialization/layout.h"
#include "complex_type.h"
#include "memory/shared_memory_types.h"
#include "serialization/serialize_information.h"

#include "serialization/debug/layout_dump.hpp"

namespace UnBARableAINS {

namespace serialization {

namespace test {

class LayoutTest : public ::testing::Test {
protected:
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

TEST_F(LayoutTest, EmptyLayout) {
    using SI = SerializeInformation<ComplexB>;
    using Vector3SI = SerializeInformation<SI::F_Tensor3::Type>;

    Layout<ComplexB> emptyLayout{};

    auto tensorLayout = emptyLayout.get(std::type_identity<SI::F_Tensor3>{}).child;

    EXPECT_NE(tensorLayout, nullptr);

    auto lengthNode = tensorLayout->get(std::type_identity<Vector3SI::F_Length>{});
    EXPECT_EQ(lengthNode.inlineSize, sizeof(size_t));
    EXPECT_EQ(lengthNode.deepSize, lengthNode.inlineSize);

    auto elementsNode = tensorLayout->get(std::type_identity<Vector3SI::F_Elements>{});
    EXPECT_EQ(elementsNode.inlineSize, 0);
    EXPECT_EQ(elementsNode.deepSize, 0);

    EXPECT_EQ(emptyLayout.getInlinedSize(), sizeof(memory::id_t));
    EXPECT_EQ(emptyLayout.getDeepSize(),
              emptyLayout.getInlinedSize() + lengthNode.deepSize + elementsNode.deepSize);
}

template <Serializable S>
class StaticLayoutUpdateTest {
public:
    StaticLayoutUpdateTest(const Layout<S>* layout)
        : m_layout(layout) {}

    template <detail::MultiField F>
    inline std::size_t getCount(void) const {
        return 2;
    }

    template <Serializable T, detail::Field F>
        requires(detail::Field<F>)
    inline auto descend(Layout<T>* childLayout, const FieldNode<F>& node) const
        -> StaticLayoutUpdateTest<T> {
        return StaticLayoutUpdateTest<T>{childLayout};
    }

    template <Serializable T, detail::MultiField F>
        requires(detail::MultiField<F>)
    inline auto descend(Layout<T>* childLayout, const FieldNode<F>& node, std::size_t index) const
        -> StaticLayoutUpdateTest<T> {
        return StaticLayoutUpdateTest<T>{childLayout};
    }

private:
    const Layout<S>* m_layout;
};

TEST_F(LayoutTest, Reconstruct) {
    Layout<ComplexB> layout{};

    EXPECT_NO_THROW(layout.update(StaticLayoutUpdateTest{&layout}););

    serialization::debug::dumpLayout(std::cout, layout, 0);
}

TEST_F(LayoutTest, EqualityOperator) {
    Layout<ComplexB> emptyLayout{};
    Layout<ComplexB> layout{data};

    // serialization::debug::dumpLayout(std::cout, emptyLayout, 0);
    // serialization::debug::dumpLayout(std::cout, layout, 0);
}

}  // namespace test

}  // namespace serialization

}  // namespace UnBARableAINS
