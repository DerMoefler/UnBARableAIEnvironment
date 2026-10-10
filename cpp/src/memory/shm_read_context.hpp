#ifndef SHM_READ_CONTEXT_H_
#define SHM_READ_CONTEXT_H_

#include <cassert>
#include <cstddef>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include "memory/shared_memory_impl.h"
#include "serialization/serialize_information.h"
#include "serialization/layout.h"
#include "utility/always_false.h"
#include "utility/typelist.h"

namespace UnBARableAINS {

namespace memory {

namespace detail {

template <serialization::detail::Fieldlike F>
using GetFieldlikeValueType_T = typename serialization::detail::GetFieldlikeValueType_MF<F>::Type;

template <typename F>
struct GetDeserializeType_MF;

template <serialization::detail::Fieldlike F>
    requires(serialization::detail::ConstructibleFromFields<GetFieldlikeValueType_T<F>>)
struct GetDeserializeType_MF<F> {
    using ValueType = GetFieldlikeValueType_T<F>;
    using Type =
        std::conditional_t<serialization::detail::MultiField<F>, std::vector<ValueType>, ValueType>;
};

template <serialization::detail::Fieldlike F>
    requires(serialization::detail::DeserializeMethodAvailable<GetFieldlikeValueType_T<F>>)
struct GetDeserializeType_MF<F> {
    using Type =
        std::conditional_t<serialization::detail::MultiField<F>, const std::span<const std::byte>,
                           typename serialization::detail::GetDeserializeDataViewType_MF<
                               GetFieldlikeValueType_T<F>>::Type>;
};

template <serialization::detail::Fieldlike... Fs>
using Fields = serialization::detail::Fields<Fs...>;

template <typename List>
struct GetDeserializableFields_MF;

template <>
struct GetDeserializableFields_MF<Fields<>> {
    using Type = Fields<>;
};

template <serialization::detail::Fieldlike Head, serialization::detail::Fieldlike... Tail>
struct GetDeserializableFields_MF<Fields<Head, Tail...>> {
    using TailTypelist = typename GetDeserializableFields_MF<Fields<Tail...>>::Type;
    using Type = std::conditional_t<
        serialization::detail::DeserializeMethodAvailable<GetFieldlikeValueType_T<Head>>,
        typename Typelist::PushFront_MF<TailTypelist, Head>::Type, TailTypelist>;
};

template <typename List>
struct GetConstructibleFields_MF;

template <>
struct GetConstructibleFields_MF<Fields<>> {
    using Type = Fields<>;
};

template <serialization::detail::Fieldlike Head, serialization::detail::Fieldlike... Tail>
struct GetConstructibleFields_MF<Fields<Head, Tail...>> {
    using TailTypelist = typename GetConstructibleFields_MF<Fields<Tail...>>::Type;
    using Type = std::conditional_t<
        serialization::detail::ConstructibleFromFields<GetFieldlikeValueType_T<Head>>,
        typename Typelist::PushFront_MF<TailTypelist, Head>::Type, TailTypelist>;
};

template <typename List>
struct GetDeserializableDataViewTypes_MF;

template <>
struct GetDeserializableDataViewTypes_MF<Fields<>> {
    using Type = Typelist::Typelist<>;
};

template <serialization::detail::Fieldlike Head, serialization::detail::Fieldlike... Tail>
struct GetDeserializableDataViewTypes_MF<Fields<Head, Tail...>> {
    using TailTypelist = typename GetDeserializableDataViewTypes_MF<Fields<Tail...>>::Type;
    using Type = std::conditional_t<
        serialization::detail::DeserializeMethodAvailable<GetFieldlikeValueType_T<Head>>,
        typename Typelist::PushFront_MF<TailTypelist,
                                        typename GetDeserializeType_MF<Head>::Type>::Type,
        TailTypelist>;
};

template <typename List>
struct GetConstructibleValueTypes_MF;

template <>
struct GetConstructibleValueTypes_MF<Fields<>> {
    using Type = Typelist::Typelist<>;
};

template <serialization::detail::Fieldlike Head, serialization::detail::Fieldlike... Tail>
struct GetConstructibleValueTypes_MF<Fields<Head, Tail...>> {
    using TailTypelist = typename GetConstructibleValueTypes_MF<Fields<Tail...>>::Type;
    using Type = std::conditional_t<
        serialization::detail::ConstructibleFromFields<GetFieldlikeValueType_T<Head>>,
        typename Typelist::PushFront_MF<TailTypelist,
                                        typename GetDeserializeType_MF<Head>::Type>::Type,
        TailTypelist>;
};

template <typename List>
struct GetNodeValueTypes_MF;

template <>
struct GetNodeValueTypes_MF<Fields<>> {
    using Type = Typelist::Typelist<>;
};

template <serialization::detail::Fieldlike Head, serialization::detail::Fieldlike... Tail>
struct GetNodeValueTypes_MF<Fields<Head, Tail...>> {
    using TailTypes = typename GetNodeValueTypes_MF<Fields<Tail...>>::Type;
    using ValueType = GetFieldlikeValueType_T<Head>;

    using NodeValueType = std::conditional_t<serialization::detail::MultiField<Head>,
                                             std::vector<ValueType>, ValueType>;

    using Type = typename Typelist::PushFront_MF<TailTypes, NodeValueType>::Type;
};

template <typename List>
struct GetDeserializableValueTypes_MF;

template <>
struct GetDeserializableValueTypes_MF<Fields<>> {
    using Type = Typelist::Typelist<>;
};

template <serialization::detail::Fieldlike Head, serialization::detail::Fieldlike... Tail>
struct GetDeserializableValueTypes_MF<Fields<Head, Tail...>> {
    using TailTypelist = typename GetDeserializableValueTypes_MF<Fields<Tail...>>::Type;

    using Type = typename Typelist::PushFront_MF<
        TailTypelist, std::conditional_t<serialization::detail::MultiField<Head>,
                                         std::vector<GetFieldlikeValueType_T<Head>>,
                                         GetFieldlikeValueType_T<Head>>>::Type;
};

namespace test {

using VSI = serialization::SerializeInformation<std::vector<int>>;
static_assert(std::same_as<typename GetDeserializableFields_MF<typename VSI::Fields>::Type,
                           Fields<VSI::F_Length, VSI::F_Elements>>,
              "std::vector<int> produces wrong DeserializableFields");
static_assert(
    std::same_as<typename GetConstructibleFields_MF<typename VSI::Fields>::Type, Fields<>>,
    "std::vector<int> produces wrong ConstructibleFields");
static_assert(std::same_as<typename GetDeserializableDataViewTypes_MF<typename VSI::Fields>::Type,
                           Typelist::Typelist<const std::span<const std::byte, 8>,
                                              const std::span<const std::byte>>>,
              "std::vector<int> produces wrong DeserializableDataViewTypes");
static_assert(std::same_as<typename GetConstructibleValueTypes_MF<typename VSI::Fields>::Type,
                           Typelist::Typelist<>>,
              "std::vector<int> produces wrong ConstructibleValueTypes");

using VVSI = serialization::SerializeInformation<std::vector<std::vector<int>>>;
static_assert(std::same_as<typename GetDeserializableFields_MF<typename VVSI::Fields>::Type,
                           Fields<VVSI::F_Length>>,
              "std::vector<std::vector<int>> produces wrong DeserializableFields");
static_assert(std::same_as<typename GetConstructibleFields_MF<typename VVSI::Fields>::Type,
                           Fields<VVSI::F_Elements>>,
              "std::vector<std::vector<int>> produces wrong ConstructibleFields");
static_assert(std::same_as<typename GetDeserializableDataViewTypes_MF<typename VVSI::Fields>::Type,
                           Typelist::Typelist<const std::span<const std::byte, 8>>>,
              "std::vector<std::vector<int>> produces wrong DeserializableDataViewTypes");
static_assert(std::same_as<typename GetConstructibleValueTypes_MF<typename VVSI::Fields>::Type,
                           Typelist::Typelist<std::vector<std::vector<int>>>>,
              "std::vector<std::vector<int>> produces wrong ConstructibleValueTypes");

}  // namespace test

}  // namespace detail

template <serialization::Serializable S, SharedMemoryImpl Shm>
class ShmReadContext {
private:
    using SI = serialization::SerializeInformation<S>;
    using Fields = typename SI::Fields;

    using NodeValueTypes = typename detail::GetNodeValueTypes_MF<Fields>::Type;

    using NodeValuesTuple = typename Typelist::Rebind_MF<std::tuple, NodeValueTypes>::Type;

    using DeserializableFields = typename detail::GetDeserializableFields_MF<Fields>::Type;
    using DeserializableValueTypes =
        typename detail::GetDeserializableValueTypes_MF<DeserializableFields>::Type;
    using DeserializablesTuple =
        typename Typelist::Rebind_MF<std::tuple, DeserializableValueTypes>::Type;

public:
    using ConstructibleFields = typename detail::GetConstructibleFields_MF<Fields>::Type;
    using ConstructibleValueTypes = typename detail::GetConstructibleValueTypes_MF<Fields>::Type;
    using ConstructiblesTuple =
        typename Typelist::Rebind_MF<std::tuple, ConstructibleValueTypes>::Type;

    inline static constexpr bool c_is_fully_deserializable =
        serialization::detail::DeserializeMethodAvailable<S>;
    inline static constexpr bool c_is_partially_deserializable =
        !(Typelist::IsEmpty_MF<DeserializableFields>::value);
    inline static constexpr std::size_t c_deserializable_fields_count =
        Typelist::Size_MF<DeserializableFields>::value;

    ShmReadContext(const serialization::Layout<S>* layout, const Shm& shmImpl,
                   ConstructiblesTuple constructedValues)
        : m_layout(layout)
        , m_shmImpl(shmImpl)
        , m_constructedValues(std::move(constructedValues)) {
        assert(m_layout);
        assert(m_layout->getSegmentId().has_value() && "Layout must have a segmentId");
    }

    S deserialize(void) const {
        assert(m_layout->getSegmentId().has_value() && "Layout must have a segmentId");
        const memory::id_t segmentId = m_layout->getSegmentId().value();
        if constexpr (c_is_fully_deserializable) {
            // TODO validate/document
            using DataView = typename serialization::detail::GetDeserializeDataViewType_MF<S>::Type;
            const std::vector<std::byte> serializedSegment = m_shmImpl.readSegment(segmentId);
            if constexpr (serialization::detail::ConstSize<S>) {
                assert(serializedSegment.size() == SI::c_serialized_size &&
                       "Read segment's size does not match the type's size (type has a constant "
                       "size here).");
            }
            return SI::deserialize(DataView{serializedSegment});
        }
        else {
            static_assert(serialization::detail::ConstructibleFromFields<S>,
                          "Serializable must be constructible from Fields here.");
            DeserializablesTuple deserializedValues = getDeserializableValues();

            NodeValuesTuple values =
                [&]<std::size_t... Is>(std::index_sequence<Is...>) -> NodeValuesTuple {
                return NodeValuesTuple{[&]() -> typename Typelist::TypeAtIndex_MF<NodeValueTypes,
                                                                                  Is>::Type {
                    using Field = typename Typelist::TypeAtIndex_MF<Fields, Is>::Type;
                    if constexpr (Typelist::Contains_MF<ConstructibleFields, Field>::value) {
                        constexpr std::size_t index =
                            Typelist::FindType_MF<ConstructibleFields, Field>::value;
                        return std::get<index>(m_constructedValues);
                    }
                    else if constexpr (Typelist::Contains_MF<DeserializableFields, Field>::value) {
                        constexpr std::size_t index =
                            Typelist::FindType_MF<DeserializableFields, Field>::value;
                        return std::get<index>(deserializedValues);
                    }
                    else {
                        static_assert(AlwaysFalse_MF<Field>::value,
                                      "Field not contained in Constructibles nor Deserializables");
                    }
                }()...};
            }(std::make_index_sequence<std::tuple_size_v<NodeValuesTuple>>{});

            return std::apply(
                []<typename... Values>(Values&&... values) -> S {
                    return SI::constructFromFields(std::forward<Values>(values)...);
                },
                std::move(values));
        }
    }

private:
    DeserializablesTuple getDeserializableValues(void) const {
        assert(m_layout->getSegmentId().has_value() && "Layout must have a segmentId");
        using DataViews =
            typename detail::GetDeserializableDataViewTypes_MF<DeserializableFields>::Type;
        static_assert(Typelist::Size_MF<DataViews>::value == c_deserializable_fields_count,
                      "Typelists must have matching sizes.");

        const memory::id_t segmentId = m_layout->getSegmentId().value();
        // TODO make this more efficient than reading the entire segment when its possibly not even
        // used
        const std::vector<std::byte> serializedSegment = m_shmImpl.readSegment(segmentId);
        memory::DataView segmentView{serializedSegment};
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) -> DeserializablesTuple {
            return DeserializablesTuple{
                [&]() -> typename Typelist::TypeAtIndex_MF<DeserializableValueTypes, Is>::Type {
                    using Field = typename Typelist::TypeAtIndex_MF<DeserializableFields, Is>::Type;
                    static_assert(serialization::FieldNode<Field>::isFullyInlined,
                                  "Field must be fully inlined");
                    using DataView = typename Typelist::TypeAtIndex_MF<DataViews, Is>::Type;
                    using DeserializableType =
                        typename Typelist::TypeAtIndex_MF<DeserializableValueTypes, Is>::Type;
                    using ValueType =
                        typename serialization::detail::GetFieldlikeValueType_MF<Field>::Type;
                    using ValueTypeSI = serialization::SerializeInformation<ValueType>;
                    const auto& node = m_layout->get(std::type_identity<Field>{});
                    static_assert(serialization::detail::DeserializeMethodAvailable<ValueType>,
                                  "ValueType must be deserializable here");
                    // TODO Implement
                    if constexpr (serialization::detail::MultiField<Field>) {
                        static_assert(serialization::detail::ConstSize<ValueType>,
                                      "Currently only supports ValueTypes of ConstSize");
                        static_assert(std::same_as<DeserializableType, std::vector<ValueType>>,
                                      "Must be same");
                        DeserializableType multiField;
                        multiField.reserve(node.count);
                        for (std::size_t i = 0; i < node.count; i++) {
                            multiField.push_back(
                                ValueTypeSI::deserialize(getView(segmentView, node, i)));
                        }
                        return multiField;
                    }
                    else if constexpr (serialization::detail::Field<Field>) {
                        return ValueTypeSI::deserialize(getView(segmentView, node, std::nullopt));
                    }
                    else {
                        static_assert(AlwaysFalse_MF<Field>::value, "Unsupported FieldType.");
                    }
                }()...};
        }(std::make_index_sequence<c_deserializable_fields_count>{});
    }

    static const auto getView(memory::DataView segmentView, const auto& node,
                              std::optional<std::size_t> index) {
        using Node = std::remove_cvref_t<decltype(node)>;
        static_assert(Node::isFullyInlined, "Node must be fully inlined");
        using Field = typename std::remove_cvref_t<Node>::Tag;
        using ValueType = typename Node::ValueType;
        using ValueTypeSI = serialization::SerializeInformation<ValueType>;
        using DataView =
            typename serialization::detail::GetDeserializeDataViewType_MF<ValueType>::Type;
        if constexpr (serialization::detail::MultiField<Field>) {
            assert(index.has_value() && "Index must provide a value for multi fields");
            static_assert(serialization::detail::ConstSize<ValueType>,
                          "ValueType must satisfy ConstSize here");
            constexpr std::size_t size = ValueTypeSI::c_serialized_size;
            return DataView{segmentView.begin() + node.offset + index.value() * size, size};
        }
        else {
            return DataView{segmentView.begin() + node.offset, node.deepSize};
        }
    }

    const serialization::Layout<S>* m_layout;
    const Shm& m_shmImpl;
    ConstructiblesTuple m_constructedValues;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SHM_READ_CONTEXT_H_
