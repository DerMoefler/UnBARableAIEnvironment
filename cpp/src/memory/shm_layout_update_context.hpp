#ifndef SHM_LAYOUT_UPDATE_CONTEXT_H_
#define SHM_LAYOUT_UPDATE_CONTEXT_H_

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "id/id_types.hpp"
#include "memory/shared_memory_types.h"
#include "memory/shared_memory_impl.h"
#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Implementation for a \ref LayoutUpdateContext using the a SharedMemoryImplementation.
 * \tparam The same Serializable as in Layout<S>::update.
 *
 * Uses the value and \ref SerializeInformation which exposes a getSize for each MultiField.
 * Uses the value obtained from get(node, value[, index]) in descend.
 */
template <serialization::Serializable S, SharedMemoryImpl Shm>
class ShmLayoutUpdateContext {
private:
    template <serialization::Serializable T>
    using Layout = serialization::Layout<T>;

    template <serialization::Serializable T>
    using SerializeInformation = serialization::SerializeInformation<T>;

    template <serialization::detail::Fieldlike T>
    using FieldNode = serialization::FieldNode<T>;

    template <serialization::detail::MultiField F>
    struct SizeHolder {
        using Tag = F;
        inline static std::size_t size = 0;
    };

    using MultiFields = typename serialization::detail::GetMultiFields_MF<S>::Type;

    template <typename List>
    struct GetTuple_MF;

    template <serialization::detail::Fieldlike... Fs>
    struct GetTuple_MF<serialization::detail::Fields<Fs...>> {
        using Type = std::tuple<SizeHolder<Fs>...>;
    };

    using SizeHolderTuple = typename GetTuple_MF<MultiFields>::Type;

public:
    /// \todo Change to not having to copy value here.
    ShmLayoutUpdateContext(Layout<S>* layout, const Shm& shmImpl, id::id_t segmentId)
        : m_layout(layout)
        , m_shmImpl(shmImpl) {
        assert(m_layout);

        layout->setSegmentId(segmentId);

        // TODO FIX
        std::vector<std::byte> serializedSegment = m_shmImpl.readSegment(segmentId);

        auto updateMultiFieldSize = [&]<serialization::detail::Fieldlike F>(SizeHolder<F>& holder) {
            using SizeField = typename F::SizeField;
            static_assert(std::same_as<typename SizeField::Type, std::size_t>,
                          "SizeField must be an std::size_t");
            const auto& sizeFieldNode = layout->get(std::type_identity<SizeField>{});
            static_assert(std::remove_cvref_t<decltype(sizeFieldNode)>::isFullyInlined,
                          "SizeFieldNode must be fully inlined.");
            const std::span sizeFieldView = {
                serializedSegment.begin() + sizeFieldNode.offset,
                serializedSegment.begin() + sizeFieldNode.offset + sizeof(std::size_t)};
            holder.size = SerializeInformation<std::size_t>::deserialize(sizeFieldView);
        };

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((updateMultiFieldSize(std::get<Is>(m_multiFieldSizes))), ...);
        }(std::make_index_sequence<std::tuple_size_v<SizeHolderTuple>>{});
    }

    template <serialization::detail::MultiField F>
    inline std::size_t getCount(void) const {
        return std::get<SizeHolder<F>>(m_multiFieldSizes).size;
    }

    template <serialization::Serializable T, serialization::detail::Fieldlike F>
        requires(serialization::detail::Field<F>)
    inline auto descend(Layout<T>* childLayout, const FieldNode<F>& node) const
        -> ShmLayoutUpdateContext<typename FieldNode<F>::ValueType, Shm> {
        return descendImpl(childLayout, node, 0);
    }

    template <serialization::Serializable T, serialization::detail::Fieldlike F>
        requires(serialization::detail::MultiField<F>)
    inline auto descend(Layout<T>* childLayout, const FieldNode<F>& node, std::size_t index) const
        -> ShmLayoutUpdateContext<typename FieldNode<F>::ValueType, Shm> {
        return descendImpl(childLayout, node, index);
    }

private:
    template <serialization::Serializable T, serialization::detail::Fieldlike F>
    auto descendImpl(Layout<T>* childLayout, const FieldNode<F>& node, std::size_t index) const
        -> ShmLayoutUpdateContext<typename FieldNode<F>::ValueType, Shm> {
        // static_assert(!serialization::FieldNode<F>::isFullyInlined,
        // "Currently unsupported for a node to be fully inlined");

        using ValueType = typename std::remove_cvref_t<decltype(node)>::ValueType;
        static_assert(std::same_as<ValueType, T>,
                      "Child Layout's ValueType does not match node's ValueType");
        constexpr std::type_identity<F> key{};

        memory::id_t parentSegmentId = m_layout->getSegmentId().value();
        memory::position_t offset{};

        if constexpr (serialization::detail::Field<F>) {
            offset = node.offset;
        }
        else {
            offset = node.offset + sizeof(memory::link_t) * index;
        }

        memory::id_t childSegmentId = m_shmImpl.getLinkedSegment(parentSegmentId, offset).value();
        return ShmLayoutUpdateContext<ValueType, Shm>{childLayout, m_shmImpl, childSegmentId};
    }

    const Layout<S>* m_layout;
    const Shm& m_shmImpl;
    SizeHolderTuple m_multiFieldSizes;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SHM_LAYOUT_UPDATE_CONTEXT_H_
