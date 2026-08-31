#pragma once
#include <cassert>
#include <iostream>  // TODO remove
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "id/id_allocator.hpp"
#include "memory/shared_memory_types.h"
#include "serialization/serialize_information.h"
#include "shared_memory_impl.h"

namespace UnBARableAINS {

namespace memory {

template <SharedMemoryImpl T, typename... SupportedTypes>
class SharedMemory {
public:
    /// \brief Variant able to hold a Layout for any of the \ref SupportedTypes.
    using LayoutVariant = std::variant<serialization::Layout<SupportedTypes>...>;

    static SharedMemory<T, SupportedTypes...> create(std::string_view name) {
        return SharedMemory<T, SupportedTypes...>(T::create(name));
    }

    static SharedMemory<T, SupportedTypes...> open(std::string_view name) {
        return SharedMemory<T, SupportedTypes...>(T::open(name));
    }

    /**
     * \brief Creates a shared memory region.
     */

    template <serialization::Serializable S>
    id::id_t write(const S& value) {
        auto layout = std::make_shared<serialization::Layout<S>>(value);

        createSegments(layout);
        writeOnCreation(value, layout);

        id::id_t serializableId = m_idAllocator.allocate();
        m_layouts.push_back(*layout);
        return serializableId;
    }

    LayoutVariant& getLayout(id::id_t serializableId) { return getLayoutVariant(serializableId); }

private:
    SharedMemory(T impl) : m_sharedMemoryImpl(impl) {}
    template <serialization::Serializable S>
    void createSegments(std::shared_ptr<serialization::Layout<S>> layout) {
        memory::id_t segmentId = m_sharedMemoryImpl.createSegment(layout->getInlinedSize());
        layout->setSegmentId(segmentId);

        auto funcBase = [segmentId](auto& _, auto& child) { child->setSegmentId(segmentId); };
        auto funcRecursive = [&](auto& _, auto& child) { createSegments(child); };

        layout->visitChildLayoutsByInlining(funcBase, funcRecursive);
    }

    template <serialization::Serializable S>
    void writeOnCreation(const S& value, std::shared_ptr<serialization::Layout<S>> layout) {
        std::cout << "SharedMemory<...>::writeOnCreation\n";
        auto writeNode = [&](auto&& self, auto& node, const auto& nodeValue) -> void {
            using Node = std::remove_cvref_t<decltype(node)>;
            using ValueType = std::remove_cvref_t<decltype(nodeValue)>;
            using SerializeInformation = serialization::SerializeInformation<ValueType>;
            using FieldTag = typename Node::Tag;
            constexpr std::type_identity<FieldTag> fieldKey{};
            // --- End of recusion ---
            if constexpr (serialization::detail::MultiField<FieldTag>) {
                using ElementFieldTag = typename FieldTag::Field;
                using ElementValueType = typename ElementFieldTag::Type;
                using ElementSerializeInformation =
                    serialization::SerializeInformation<ElementValueType>;
                std::cout << typeid(ElementValueType).name() << ":" << typeid(ValueType).name()
                          << "\n";
                if constexpr (serialization::detail::SerializeMethodAvailable<ElementValueType>) {
                    std::cout << "SharedMemory<...>::writeOnCreation::writeNode: Base (Multi)\n";
                    size_t segmentId = layout->getSegmentId().value();
                    for (int i = 0; i < node.count; i++) {
                        auto element = SerializeInformation::get(fieldKey, nodeValue, i);
                        auto serialized = ElementSerializeInformation::serialize(element);
                        try {
                            m_sharedMemoryImpl.appendToSegment(segmentId, serialized);
                        } catch (const std::exception& e) {
                            std::cerr << e.what() << "\n";
                        }
                    }
                }
                else {
                    std::cout << "SharedMemory<...>::writeOnCreation::writeNode: Recurse (Multi)\n";
                    for (int i = 0; i < node.count; i++) {
                        auto element = SerializeInformation::get(fieldKey, nodeValue, i);
                        serialization::visitNodeChildLayouts(node, [&](auto& childLayout) {
                            childLayout->visitNodesByInlining(
                                [&](auto& nestedNode) {
                                    self(self, nestedNode,
                                         SerializeInformation::get(fieldKey, nodeValue));
                                },
                                [](auto& nestedNode) {
                                    // static_assert(AlwaysFalse_MF<S>::value, "ERROR");
                                    assert(false && "Assertion failed at runtime");
                                });
                        });
                    }
                }
            }
            else if constexpr (serialization::detail::SerializeMethodAvailable<ValueType>) {
                std::cout << "SharedMemory<...>::writeOnCreation::writeNode: Base (Single)"
                          << std::endl;
                size_t segmentId = layout->getSegmentId().value();
                auto serialized = SerializeInformation::serialize(nodeValue);
                try {
                    m_sharedMemoryImpl.appendToSegment(segmentId, serialized);
                } catch (const std::exception& e) {
                    std::cerr << e.what();
                }
            }
            else {
                std::cout << "SharedMemory<...>::writeOnCreation::writeNode: Recurse (Single)"
                          << std::endl;
                serialization::visitNodeChildLayouts(node, [&](auto& childLayout) {
                    childLayout->visitNodesByInlining(
                        [&](auto& nestedNode) {
                            self(self, nestedNode, SerializeInformation::get(fieldKey, nodeValue));
                        },
                        [](auto& nestedNode) {
                            // static_assert(AlwaysFalse_MF<S>::value, "ERROR");
                            assert(false && "Assertion failed at runtime");
                        });
                });
            }
        };
        auto funcBase = [&](auto& node) { writeNode(writeNode, node, value); };
        auto funcRecursive = [&](auto& node) {
            using SerializeInformation = serialization::SerializeInformation<S>;
            using Node = std::remove_cvref_t<decltype(node)>;
            using FieldTag = typename Node::Tag;
            constexpr std::type_identity<FieldTag> fieldKey{};

            if constexpr (serialization::detail::MultiField<FieldTag>) {
                for (size_t i = 0; i < node.count; i++) {
                    assert(node.children[i]);  // cannot be nullptr
                    writeOnCreation(SerializeInformation::get(fieldKey, value, i),
                                    node.children[i]);
                }
            }
            else {
                assert(node.child);  // cannot be nullptr
                writeOnCreation(SerializeInformation::get(fieldKey, value), node.child);
            }
        };

        layout->visitNodesByInlining(funcBase, funcRecursive);
    }

    LayoutVariant& getLayoutVariant(id::id_t serializableId) {
        // TODO currently no deletion possible, change when implemented
        if (serializableId >= m_layouts.size())
            throw std::out_of_range("Serializable ID out of range");
        return m_layouts[serializableId];
    }

    std::vector<LayoutVariant> m_layouts;
    id::IdAllocator m_idAllocator;

    T m_sharedMemoryImpl;
};

};  // namespace memory

};  // namespace UnBARableAINS
