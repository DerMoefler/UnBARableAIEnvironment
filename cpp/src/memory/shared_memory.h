#pragma once
#include <cassert>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "id/id_types.hpp"
#include "id/id_allocator.hpp"
#include "memory/shared_memory_impl.h"
#include "memory/shared_memory_types.h"
#include "memory/shm_layout_update_context.hpp"
#include "memory/shm_read_context.hpp"
#include "serialization/serialize_information.h"
#include "serialization/layout.h"
#include "utility/always_false.h"
#include "utility/type_index.hpp"
#include "utility/typelist.h"

namespace UnBARableAINS {

namespace memory {

namespace detail {
inline auto shmLogger = spdlog::stdout_color_mt("SharedMemory");
}

/**
 * \brief A higher level view on the shared-memory using a SharedMemoryImpl \p T to interact with a
 * physical shared-memory region.
 *
 * This is essentially a Type view on the shared-memory while the SharedMemoryImpl was a byte-view.
 */
template <SharedMemoryImpl T, serialization::Serializable... SupportedTypes>
class SharedMemory {
public:
    /// \brief Typelist holding \ref SupportedTypes.
    using SupportedTypelist = Typelist::Typelist<SupportedTypes...>;

    /// \brief Variant able to hold a Layout for any of the \ref SupportedTypes.
    using LayoutVariant = std::variant<serialization::Layout<SupportedTypes>...>;

    /// \brief Variant able to hold one of the \ref SupportedTypes.
    using ValueVariant = std::variant<SupportedTypes...>;

    /// \brief Templated type alias for the UpdateContext.
    template <serialization::Serializable S>
    using ShmContext = ShmLayoutUpdateContext<S, T>;

    /// \brief Simple value template for getting the type index into supported types (starts at 1,
    /// not 0).
    template <serialization::Serializable S>
    inline static constexpr std::size_t c_type_index =
        TypeIndex_MF<S, SupportedTypes...>::index + 1;

    /**
     * \brief Creates a new shared-memory region using the implementation \ref T.
     * \param name Name of the shared-memory region (should start with a '/' per convention).
     * \param initialSerializableTableSize Initial number of elements in the SerializableTable.
     * \return A SharedMemory object.
     */
    inline static SharedMemory<T, SupportedTypes...> create(
        const std::string_view name, const std::size_t initialSerializableTableSize) {
        SharedMemory<T, SupportedTypes...> shm{T::create(name), initialSerializableTableSize};
        shm.createSerializableTable();
        SPDLOG_LOGGER_INFO(detail::shmLogger,
                           "Created shared-memory with capacity for {} segments.",
                           initialSerializableTableSize);
        return shm;
    }

    /**
     * \brief Opens an existing shared-memory region.
     * \param name Name of the shared-memory region to open.
     * \return An initialized SharedMemory object.
     * \throws std::system_error if the shared-memory region cannot be opened, see \ref
     * SegmentTable::getSegmentInformation() if c_serializable_table_segment_id does not exist, see
     * \ref update if the table if ill formed.
     *
     * This method does an update() internally, meaning the Layouts are initialized according to the
     * current state of the shm.
     */
    inline static SharedMemory<T, SupportedTypes...> open(const std::string_view name) {
        // TODO the calculation for the initial size has to be refactores once extending is
        // possible. This is because extension will create multiple segments in T, while the initial
        // size is the size of one of such segments.
        auto impl = T::open(name);
        const std::size_t serializableTableSize =
            impl.getSegmentInformation(c_serializable_table_segment_id).getHeader().getDataSize();
        const std::size_t numElements = serializableTableSize / c_serializable_table_entry_size;

        SharedMemory<T, SupportedTypes...> shm{std::move(impl), numElements};
        shm.update();
        SPDLOG_LOGGER_INFO(detail::shmLogger, "Opened shared-memory with capacity for {} segments.",
                           numElements);
        return shm;
    }

    /**
     * \brief Remove a shared-memory region.
     * \param name Name of the shared-memory to remove.
     * \throws std::system_error if the shared-memory region cannot be opened
     */
    inline static void remove(const std::string& name) { T::remove(name); }

    /**
     * \ref Update the information about the shared-memory region.
     * \returns Whether or not any layout changes have been detected.
     * \note Value changed are not reported, only changes in size.
     */
    bool update(void) {
        bool change = m_sharedMemoryImpl.update();

        std::vector<id::id_t> serializableIds;
        change |= updateLayouts(serializableIds);

        const auto oldAllocator = m_idAllocator;
        m_idAllocator = id::IdAllocator();
        m_idAllocator.setUsedIds(
            std::set<id::id_t>(serializableIds.begin(), serializableIds.end()));
        change |= oldAllocator != m_idAllocator;
        return change;
    }

    /**
     * \brief Create and write a new Serializable to the shared-memory region.
     * \tparam S A Serializable contained in \ref SupportedTypes.
     * \param value A value of Type \ref S.
     * \returns The serializableId assigned to the written \p value.
     */
    template <serialization::Serializable S>
    id::id_t write(const S& value) {
        if (serializablesCount() == m_serializableTableSize) {
            throw std::runtime_error("SharedMemory currently only supports " +
                                     std::to_string(m_serializableTableSize) + " elements.");
        }
        auto layout = serialization::Layout<S>(value);

        createSegments(&layout);
        // serialization::debug::dumpLayout(std::cout, layout, 0);
        writeOnCreation(value, &layout);

        id::id_t serializableId = m_idAllocator.allocate();
        createSerializableTableEntry<S>(serializableId, layout.getSegmentId().value());
        m_layouts.push_back(std::move(layout));
        // std::cout << "SharedMemory<Impl, SupportedTypes...>::write: Finished successfully" <<
        // std::endl;
        return serializableId;
    }

    /**
     * \brief Read an existing Serializable.
     * \param serializableId Id of the Serializable, assigned previously by \ref write.
     * \returns A \ref ValueVariant containing the read value.
     * \throws std::out_of_range if serializableId is outside of the range.
     */
    ValueVariant read(id::id_t serializableId) const {
        // std::cout << "SharedMemory::read\n";
        const LayoutVariant& layout = getLayoutVariant(serializableId);

        return std::visit([&](auto&& layout) -> ValueVariant { return readImpl(&layout); }, layout);
    }

    /**
     * \brief Get the number of Serializables in the shared-memory.
     * \returns Number of Serializables.
     */
    std::size_t serializablesCount(void) const { return m_layouts.size(); }

    /**
     * \brief Get the Layout for an existing Serializable.
     * \param serializableId Id of the Serializable, assigned previously by \ref write.
     * \returns A \ref LayoutVariant containing the Layout.
     * \throws std::out_of_range if serializableId is outside of the range.
     */
    LayoutVariant& getLayout(id::id_t serializableId) { return getLayoutVariant(serializableId); }

private:
    /// \brief Constant id for the SerializableTable.
    static constexpr memory::id_t c_serializable_table_segment_id = 0;
    /// \brief Size of an entry in the SerializableTable.
    inline static constexpr std::size_t c_serializable_table_entry_size = 3 * sizeof(id::id_t);

    SharedMemory(T impl, const std::size_t initialSerializableTableSize)
        : m_sharedMemoryImpl(std::move(impl))
        , m_serializableTableSize(initialSerializableTableSize) {}

    /**
     * \brief Creates the SerializableTable with segmentId \ref c_serializable_table_segment_id in
     * shared-memory.
     *
     * \pre Shared-memory is completely empty.
     * \throws std::runtime_error if shared-memory not empty or assigned segmentId not = 0.
     *
     */
    void createSerializableTable(void) {
        const std::size_t segmentsCount = m_sharedMemoryImpl.getSegmentsCount();
        if (segmentsCount) {
            const std::string message = "Shared-memory region not empty.";
            SPDLOG_LOGGER_ERROR(detail::shmLogger, message);
            throw std::runtime_error(message);
        }
        memory::id_t layoutTableSegmentId = m_sharedMemoryImpl.createSegment(
            m_serializableTableSize * c_serializable_table_entry_size);
        if (layoutTableSegmentId != c_serializable_table_segment_id) {
            const std::string message =
                "The first (0th) element must be available for the layout table";
            SPDLOG_LOGGER_ERROR(detail::shmLogger, message);
            throw std::runtime_error(message);
        }
    }

    /**
     * \brief Create an entry into the SerializableTable.
     * \param serializableId Id of the Serializable.
     * \param mainSegmentId Id assigned by the implementaiton \p T for the root element.
     * \todo has to be refactored, this is just to test out the idea / get up and running.
     *
     * Looks up the typeIndex using \ref c_type_index.
     */
    template <serialization::Serializable S>
    void createSerializableTableEntry(id::id_t serializableId, id::id_t mainSegmentId) {
        constexpr memory::id_t index = static_cast<id::id_t>(c_type_index<S>);
        using SerializeInformation = serialization::SerializeInformation<id::id_t>;
        m_sharedMemoryImpl.appendToSegment(c_serializable_table_segment_id,
                                           SerializeInformation::serialize(serializableId));
        m_sharedMemoryImpl.appendToSegment(c_serializable_table_segment_id,
                                           SerializeInformation::serialize(mainSegmentId));
        m_sharedMemoryImpl.appendToSegment(c_serializable_table_segment_id,
                                           SerializeInformation::serialize(index));
        SPDLOG_LOGGER_TRACE(
            detail::shmLogger,
            "Entry for SerializableID: {} with mainSegmentId {} and typeIndex {} created.",
            serializableId, mainSegmentId, index);
    }

    /**
     * \brief Updates the Layouts present in the shared-memory.
     * \param[out] SerializableIds found in the shared-memory (must be empty on call).
     * \returns Whether any changed to the layouts have been detected.
     * \throws std::runtime_error if the SerializedTable is ill formed.
     */
    bool updateLayouts(std::vector<id::id_t>& serializableIds) {
        assert(serializableIds.size() == 0 && "Function expects an empty serializableIds");
        const auto oldLayouts = m_layouts;

        std::vector<std::byte> serializedLayoutTable =
            m_sharedMemoryImpl.readSegment(c_serializable_table_segment_id);
        if (serializedLayoutTable.size() % c_serializable_table_entry_size) {
            const std::string message =
                "Ill formed SerializableTable. Written bytes must be a multiple of "
                "c_serializable_table_entry_size";
            SPDLOG_LOGGER_ERROR(detail::shmLogger, message);
            throw std::runtime_error(message);
        }
        m_layouts.clear();
        const std::size_t numEntries =
            serializedLayoutTable.size() / c_serializable_table_entry_size;
        m_layouts.reserve(numEntries);
        serializableIds.reserve(numEntries);
        using IdSI = serialization::SerializeInformation<id::id_t>;
        for (std::size_t i = 0; i < numEntries; i++) {
            const std::span entryView{
                serializedLayoutTable.begin() + i * c_serializable_table_entry_size,
                serializedLayoutTable.begin() + i * c_serializable_table_entry_size +
                    c_serializable_table_entry_size};

            id::id_t serializableId =
                IdSI::deserialize(entryView.template subspan<0, sizeof(id::id_t)>());
            id::id_t mainSegmentId =
                IdSI::deserialize(entryView.template subspan<sizeof(id::id_t), sizeof(id::id_t)>());
            id::id_t typeIndex = IdSI::deserialize(
                entryView.template subspan<2 * sizeof(id::id_t), sizeof(id::id_t)>());
            // std::cout << "SharedMemory<...>::initializeLayouts: " << serializableId << ":"
            //          << mainSegmentId << ":" << typeIndex << "\n";
            LayoutVariant layout = buildLayout(mainSegmentId, typeIndex);
            m_layouts.push_back(std::move(layout));
            serializableIds.push_back(serializableId);
        }
        return oldLayouts != m_layouts;
    }

    /**
     * \brief Read and build a new Layout for the Serializable.
     * \tparam NTTP index used to compare against \p typeIndex and recurse until they either match
     * or we find the Type to be unsupported.
     * \param mainSegmentId Id assigned by the SharedMemoryImpl \p T.
     * \param typeIndex Index of the Serializable's Type into SupportedTypes.
     */
    template <id::id_t i = 0>
    LayoutVariant buildLayout(const id::id_t mainSegmentId, const id::id_t typeIndex) {
        if (typeIndex == i + 1) {
            using Serializable = std::tuple_element_t<i, std::tuple<SupportedTypes...>>;
            Serializable value{};
            serialization::Layout<Serializable> layout{};
            layout.template update<ShmContext>(ShmLayoutUpdateContext<Serializable, T>{
                &layout, m_sharedMemoryImpl, mainSegmentId});
            return layout;
        }
        if constexpr (i + 1 < sizeof...(SupportedTypes)) {
            return buildLayout<i + 1>(mainSegmentId, typeIndex);
        }
        else {
            throw std::runtime_error("Invalid type index in layout table.");
        }
    }

    /**
     * \brief Create the segments in the physical shared-memory for the computed \p layout.
     * \tparam S A supported Serializable.
     * \param layout Pointer to the layout.
     *
     * This method work recursively by using Layout::visitChildLayoutsByInlining. It starts by
     * creating a segment for the \p layout with its inlinedSize. Then, it visits each child layout
     * and either creates a segment for a fully inlined child or recurses using this method.
     *
     * \note The pointer syntax is only required because a FieldNode provides a std::shared_ptr for
     * its layout(s). If instead it just stored it as a non-pointer, this could be refactored.
     */
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

    template <serialization::detail::FundamentalSerializable S>
    void writeOnCreation(const S& value, serialization::Layout<S>* layout) {
        std::size_t segmentId = layout->getSegmentId().value();
        auto serialized = serialization::SerializeInformation<S>::serialize(value);
        m_sharedMemoryImpl.appendToSegment(segmentId, serialized);
        return;
    }

    template <serialization::Serializable S>
        requires(not serialization::detail::FundamentalSerializable<S>)
    void writeOnCreation(const S& value, serialization::Layout<S>* layout) {
        // std::cout << "SharedMemory<...>::writeOnCreation\n";
        size_t segmentId = layout->getSegmentId().value();

        auto linkChildSegment = [&](memory::id_t childSegmentId) {
            auto serialized =
                serialization::SerializeInformation<memory::id_t>::serialize(childSegmentId);
            m_sharedMemoryImpl.appendToSegment(segmentId, serialized);
        };

        auto funcBase = [&](const auto& node) {
            using Node = std::remove_cvref_t<decltype(node)>;
            // std::cout << "Base (Inlining) for field \""
            //           << serialization::debug::displayName<typename Node::Tag>() << "\"\n";
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
                    // std::cout << "Writing " << serialized.size() << " bytes into segment "
                    //           << segmentId << "\n";
                    m_sharedMemoryImpl.appendToSegment(segmentId, serialized);
                }
            };
            serialization::visitValueFields(node, value, func);
        };

        auto funcRecursive = [&](const auto& node) {
            using Node = std::remove_cvref_t<decltype(node)>;
            // std::cout << "Recursing (Inlining) for field \""
            // << serialization::debug::displayName<typename Node::Tag>() << "\"\n";
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

    template <serialization::Serializable S>
    S readImpl(const serialization::Layout<S>* layout) const {
        using Layout = serialization::Layout<S>;
        using SI = typename Layout::SI;
        using ValueType = typename SI::Type;
        using Context = ShmReadContext<ValueType, T>;

        if constexpr (Context::c_is_fully_deserializable) {
            // std::cout << "SharedMemory::readImpl: Fully deserializable\n";
            Context context{layout, m_sharedMemoryImpl, {}};
            return context.deserialize();
        }
        else {
            // std::cout << "SharedMemory::readImpl: Not fully deserializable (recursing)\n";
            using ConstructibleFields = typename Context::ConstructibleFields;
            using ConstructibleValueTypes = typename Context::ConstructibleValueTypes;
            using ConstructiblesTuple = typename Context::ConstructiblesTuple;

            auto constructedValues =
                [&]<std::size_t... Is>(std::index_sequence<Is...>) -> ConstructiblesTuple {
                return ConstructiblesTuple{
                    [&]() -> typename Typelist::TypeAtIndex_MF<ConstructibleValueTypes, Is>::Type {
                        using Field =
                            typename Typelist::TypeAtIndex_MF<ConstructibleFields, Is>::Type;
                        using ConstructibleValueType =
                            typename Typelist::TypeAtIndex_MF<ConstructibleValueTypes, Is>::Type;
                        const auto& node = layout->get(std::type_identity<Field>{});
                        if constexpr (serialization::detail::MultiField<Field>) {
                            ConstructibleValueType multiFieldValues;
                            static_assert(std::same_as<typename ConstructibleValueType::value_type,
                                                       typename serialization::detail::
                                                           GetFieldlikeValueType_MF<Field>::Type>,
                                          "Error");
                            multiFieldValues.reserve(node.count);
                            const auto& children = node.children;
                            for (const auto& child : children) {
                                assert(child && "Child canot be nullptr");
                                multiFieldValues.push_back(readImpl(child.get()));
                            }
                            return multiFieldValues;
                        }
                        else if constexpr (serialization::detail::Field<Field>) {
                            assert(node.child && "Child cannot be nullptr");
                            return readImpl(node.child.get());
                        }
                        else {
                            static_assert(AlwaysFalse_MF<Field>::value, "Unsupported FieldType.");
                        }
                    }()...};
            }(std::make_index_sequence<std::tuple_size_v<ConstructiblesTuple>>{});

            Context context{layout, m_sharedMemoryImpl, constructedValues};
            return context.deserialize();
        }
    }

    const LayoutVariant& getLayoutVariant(id::id_t serializableId) const {
        // TODO currently no deletion possible, change when implemented
        if (serializableId >= m_layouts.size())
            throw std::out_of_range("Serializable ID out of range");
        return m_layouts[serializableId];
    }

    LayoutVariant& getLayoutVariant(id::id_t serializableId) {
        // TODO currently no deletion possible, change when implemented
        if (serializableId >= m_layouts.size())
            throw std::out_of_range("Serializable ID out of range");
        return m_layouts[serializableId];
    }

    /// \todo This currently cannot change because the segment cant grow, fix it.
    std::size_t m_serializableTableSize;
    std::vector<LayoutVariant> m_layouts;
    id::IdAllocator m_idAllocator;

    T m_sharedMemoryImpl;
};

};  // namespace memory

};  // namespace UnBARableAINS
