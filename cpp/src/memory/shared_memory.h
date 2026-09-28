#pragma once
#include <cassert>
#include <iostream>  // TODO remove
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "id/id_types.hpp"
#include "serialization/debug/type_name.hpp"  // TODO remove
#include "serialization/debug/layout_dump.hpp"
#include "id/id_allocator.hpp"
#include "memory/shared_memory_types.h"
#include "memory/shm_layout_update_context.hpp"
#include "serialization/serialize_information.h"
#include "serialization/layout.h"
#include "shared_memory_impl.h"
#include "utility/always_false.h"
#include "utility/type_index.hpp"

namespace UnBARableAINS {

namespace memory {

template <SharedMemoryImpl T, serialization::Serializable... SupportedTypes>
class SharedMemory {
public:
    /// \brief Variant able to hold a Layout for any of the \ref SupportedTypes.
    using LayoutVariant = std::variant<serialization::Layout<SupportedTypes>...>;

    template <serialization::Serializable S>
    using ShmContext = ShmLayoutUpdateContext<S, T>;

    /// \brief Simple value template for getting the type index into supported types (starts at 1,
    /// not 0).
    template <serialization::Serializable S>
    inline static constexpr std::size_t c_type_index =
        TypeIndex_MF<S, SupportedTypes...>::index + 1;

    inline static SharedMemory<T, SupportedTypes...> create(std::string_view name) {
        SharedMemory<T, SupportedTypes...> shm(T::create(name));
        shm.createLayoutTable();
        return shm;
    }

    inline static SharedMemory<T, SupportedTypes...> open(std::string_view name) {
        SharedMemory<T, SupportedTypes...> shm(T::open(name));
        shm.initializeLayouts();
        return shm;
    }

    inline static void remove(const std::string& name) { T::remove(name); }

    template <serialization::Serializable S>
    id::id_t write(const S& value) {
        if (m_idAllocator.size() == c_layout_table_size_increase) {
            throw std::runtime_error("SharedMemory currently only supports " +
                                     std::to_string(c_layout_table_size_increase) + " elements.");
        }
        auto layout = serialization::Layout<S>(value);

        createSegments(&layout);
        serialization::debug::dumpLayout(std::cout, layout, 0);
        writeOnCreation(value, &layout);

        id::id_t serializableId = m_idAllocator.allocate();
        createLayoutTableEntry<S>(serializableId, layout.getSegmentId().value());
        m_layouts.push_back(std::move(layout));
        return serializableId;
    }

    LayoutVariant& getLayout(id::id_t serializableId) { return getLayoutVariant(serializableId); }

private:
    inline static constexpr memory::id_t c_layout_table_segment_id = 0;
    inline static constexpr std::size_t c_layout_table_size_increase = 8;
    inline static constexpr std::size_t c_layout_table_entry_size = 3 * sizeof(id::id_t);
    inline static constexpr std::size_t c_partial_layout_table_size =
        c_layout_table_size_increase * c_layout_table_entry_size;

    SharedMemory(T impl)
        : m_sharedMemoryImpl(std::move(impl)) {}

    void createLayoutTable(void) {
        memory::id_t layoutTableSegmentId =
            m_sharedMemoryImpl.createSegment(c_partial_layout_table_size);
        assert(layoutTableSegmentId == c_layout_table_segment_id &&
               "The first (0th) element must be available for the layout table");
    }

    /// \todo has to be refactored, this is just to test out the idea / get up and running.
    template <serialization::Serializable S>
    void createLayoutTableEntry(id::id_t serializableId, id::id_t mainSegmentId) {
        constexpr memory::id_t index = static_cast<id::id_t>(c_type_index<S>);
        using SerializeInformation = serialization::SerializeInformation<id::id_t>;
        m_sharedMemoryImpl.appendToSegment(c_layout_table_segment_id,
                                           SerializeInformation::serialize(serializableId));
        m_sharedMemoryImpl.appendToSegment(c_layout_table_segment_id,
                                           SerializeInformation::serialize(mainSegmentId));
        m_sharedMemoryImpl.appendToSegment(c_layout_table_segment_id,
                                           SerializeInformation::serialize(index));
        std::cout << "SharedMemory::createLayoutTableEntry: created with SerializableID: "
                  << serializableId << ", mainSegmentId: " << mainSegmentId << "\n";
    }

    void initializeLayouts(void) {
        assert(m_idAllocator.size() == 0 && "Can only initialize for an empty shm");
        std::vector<std::byte> serializedLayoutTable =
            m_sharedMemoryImpl.readSegment(c_layout_table_segment_id);
        assert((serializedLayoutTable.size() % c_layout_table_entry_size) == 0 &&
               "Ill formed layout table.");
        std::size_t numEntries = serializedLayoutTable.size() / c_layout_table_entry_size;
        using SerializeInformation = serialization::SerializeInformation<id::id_t>;
        for (std::size_t i = 0; i < numEntries; i++) {
            const std::span entryView{serializedLayoutTable.begin() + i * c_layout_table_entry_size,
                                      serializedLayoutTable.begin() +
                                          i * c_layout_table_entry_size +
                                          c_layout_table_entry_size};

            id::id_t serializableId = SerializeInformation::deserialize(
                entryView.template subspan<0, sizeof(id::id_t)>());
            id::id_t mainSegmentId = SerializeInformation::deserialize(
                entryView.template subspan<sizeof(id::id_t), sizeof(id::id_t)>());
            id::id_t typeIndex = SerializeInformation::deserialize(
                entryView.template subspan<2 * sizeof(id::id_t), sizeof(id::id_t)>());
            std::cout << "SharedMemory<...>::initializeLayouts: " << serializableId << ":"
                      << mainSegmentId << ":" << typeIndex << "\n";
            LayoutVariant layout = buildLayout(serializableId, mainSegmentId, typeIndex);
            m_layouts.push_back(std::move(layout));
        }
    }

    template <id::id_t i = 0>
    LayoutVariant buildLayout(id::id_t serializableId, id::id_t mainSegmentId, id::id_t typeIndex) {
        if (typeIndex == i + 1) {
            using Serializable = std::tuple_element_t<i, std::tuple<SupportedTypes...>>;
            Serializable value{};
            serialization::Layout<Serializable> layout{};
            layout.template update<ShmContext>(ShmLayoutUpdateContext<Serializable, T>{
                &layout, m_sharedMemoryImpl, mainSegmentId});
            return layout;
        }
        if constexpr (i + 1 < sizeof...(SupportedTypes)) {
            return buildLayout<i + 1>(serializableId, mainSegmentId, typeIndex);
        }
        else {
            throw std::runtime_error("Invalid type index in layout table.");
        }
    }

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

        auto linkChildSegment = [&](memory::id_t childSegmentId) {
            auto serialized =
                serialization::SerializeInformation<memory::id_t>::serialize(childSegmentId);
            m_sharedMemoryImpl.appendToSegment(segmentId, serialized);
        };

        auto funcBase = [&](const auto& node) {
            using Node = std::remove_cvref_t<decltype(node)>;
            std::cout << "Base (Inlining) for field \""
                      << serialization::debug::displayName<typename Node::Tag>() << "\"\n";
            auto func = [&](const auto& value) {
                using ValueType = std::remove_cvref_t<decltype(value)>;
                if constexpr (!serialization::detail::SerializeMethodAvailable<ValueType>) {
                    static_assert(AlwaysFalse_MF<ValueType>::value,
                                  "Cannot serialize ValueType. Probably trying to inline a type, "
                                  "that itselfs inlines other types.");
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
                    memory::id_t childSegmentId = node.children[i]->getSegmentId().value();
                    linkChildSegment(childSegmentId);
                }
            }
            else {
                assert(node.child.get());  // cannot be nullptr
                writeOnCreation(SerializeInformation::get(fieldKey, value), node.child.get());
                memory::id_t childSegmentId = node.child->getSegmentId().value();
                linkChildSegment(childSegmentId);
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
