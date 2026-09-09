#pragma once
#include <cassert>
#include <iostream>  // TODO remove
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "serialization/debug/type_name.hpp"  // TODO remove
#include "serialization/debug/layout_dump.hpp"
#include "id/id_allocator.hpp"
#include "memory/shared_memory_types.h"
#include "serialization/serialize_information.h"
#include "serialization/layout.h"
#include "shared_memory_impl.h"
#include "utility/always_false.h"

namespace UnBARableAINS {

namespace memory {

template <SharedMemoryImpl T, typename... SupportedTypes>
class SharedMemory {
public:
    /// \brief Variant able to hold a Layout for any of the \ref SupportedTypes.
    using LayoutVariant = std::variant<serialization::Layout<SupportedTypes>...>;

    inline static SharedMemory<T, SupportedTypes...> create(std::string_view name) {
        return SharedMemory<T, SupportedTypes...>(T::create(name));
    }

    inline static SharedMemory<T, SupportedTypes...> open(std::string_view name) {
        return SharedMemory<T, SupportedTypes...>(T::open(name));
    }

    inline static void remove(const std::string& name) { T::remove(name); }

    /**
     * \brief Creates a shared memory region.
     */

    template <serialization::Serializable S>
    id::id_t write(const S& value) {
        auto layout = serialization::Layout<S>(value);

        createSegments(&layout);
        serialization::debug::dumpLayout(std::cout, layout, 0);
        writeOnCreation(value, &layout);

        id::id_t serializableId = m_idAllocator.allocate();
        m_layouts.push_back(std::move(layout));
        return serializableId;
    }

    LayoutVariant& getLayout(id::id_t serializableId) { return getLayoutVariant(serializableId); }

private:
    SharedMemory(T impl)
        : m_sharedMemoryImpl(std::move(impl)) {}
    template <serialization::Serializable S>
    void createSegments(serialization::Layout<S>* layout) {
        memory::id_t segmentId = m_sharedMemoryImpl.createSegment(layout->getInlinedSize());
        layout->setSegmentId(segmentId);

        auto funcBase = [&](auto& _, auto& child) {
            memory::id_t childSegmentId = m_sharedMemoryImpl.createSegment(child->getInlinedSize());
            child->setSegmentId(childSegmentId);
        };
        auto funcRecursive = [&](auto& _, auto& child) { createSegments(child.get()); };

        layout->visitChildLayoutsByInlining(funcBase, funcRecursive);
    }

    template <serialization::Serializable S>
    void writeOnCreation(const S& value, serialization::Layout<S>* layout) {
        std::cout << "SharedMemory<...>::writeOnCreation\n";
        size_t segmentId = layout->getSegmentId().value();

        auto funcBase = [&](const auto& node) {
            using Node = std::remove_cvref_t<decltype(node)>;
            std::cout << "Base (Inlining) for field \""
                      << serialization::debug::displayName<typename Node::Tag>() << "\"\n";
            auto func = [&](const auto& value) {
                using ValueType = std::remove_cvref_t<decltype(value)>;
                if constexpr (!serialization::detail::SerializeMethodAvailable<ValueType>) {
                    static_assert(AlwaysFalse_MF<ValueType>::value,
                                  "Cannot serialize ValueType. Probably trying to inline a type, "
                                  "that itselfs inline other types.");
                }
                else {
                    auto serialized =
                        serialization::SerializeInformation<ValueType>::serialize(value);
                    std::cout << "Writing " << serialized.size() << " bytes into segment "
                              << segmentId << "\n";
                    m_sharedMemoryImpl.appendToSegment(segmentId, serialized);
                }
            };
            serialization::visitValueFields(node, value, func);
        };

        auto funcRecursive = [&](const auto& node) {
            using Node = std::remove_cvref_t<decltype(node)>;
            std::cout << "Recursing (Inlining) for field \""
                      << serialization::debug::displayName<typename Node::Tag>() << "\"\n";
            using SerializeInformation = serialization::SerializeInformation<S>;
            using Node = std::remove_cvref_t<decltype(node)>;
            using FieldTag = typename Node::Tag;
            constexpr std::type_identity<FieldTag> fieldKey{};

            if constexpr (serialization::detail::MultiField<FieldTag>) {
                for (size_t i = 0; i < node.count; i++) {
                    assert(node.children[i]);  // cannot be nullptr
                    writeOnCreation(SerializeInformation::get(fieldKey, value, i),
                                    node.children[i].get());
                }
            }
            else {
                assert(node.child.get());  // cannot be nullptr
                writeOnCreation(SerializeInformation::get(fieldKey, value), node.child.get());
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
