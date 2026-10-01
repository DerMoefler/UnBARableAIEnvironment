#ifndef SHM_READ_CONTEXT_H_
#define SHM_READ_CONTEXT_H_

#include <cstddef>
#include <span>
#include <tuple>
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
    using Type = GetFieldlikeValueType_T<F>;
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
                           Typelist::Typelist<std::vector<int>>>,
              "std::vector<std::vector<int>> produces wrong ConstructibleValueTypes");

}  // namespace test

}  // namespace detail

template <serialization::Serializable S, SharedMemoryImpl Shm>
class ShmReadContext {
private:
    using SI = serialization::SerializeInformation<S>;
    using Fields = typename SI::Fields;

    using NodeValueTypes = typename serialization::detail::GetFieldsValueTypelist_MF<Fields>::Type;

    using NodeValuesTuple = typename Typelist::Rebind_MF<std::tuple, NodeValueTypes>::Type;

    using DeserializableFields = typename detail::GetDeserializableFields_MF<Fields>::Type;
    using DeserializableValueTypes =
        typename serialization::detail::GetFieldsValueTypelist_MF<DeserializableFields>::Type;
    using DeserializablesTuple =
        typename Typelist::Rebind_MF<std::tuple, DeserializableValueTypes>::Type;

    using ConstructibleFields = typename detail::GetConstructibleFields_MF<Fields>::Type;
    using ConstructibleValueTypes = typename detail::GetConstructibleValueTypes_MF<Fields>::Type;
    using ConstructiblesTuple =
        typename Typelist::Rebind_MF<std::tuple, ConstructibleValueTypes>::Type;

public:
    inline static constexpr bool c_is_fully_deserializable =
        serialization::detail::DeserializeMethodAvailable<S>;
    inline static constexpr bool c_is_partially_deserializable =
        Typelist::IsEmpty_MF<DeserializableFields>::value;
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
            return SI::deserialize(DataView{serializedSegment});
        }
        else {
            static_assert(serialization::detail::ConstructibleFromFields<S>,
                          "Serializable must be constructible from Fields here.");
            DeserializablesTuple deserializedValues{getDeserializableValues()};

            NodeValuesTuple values =
                [&]<std::size_t... Is>(std::index_sequence<Is...>) -> NodeValuesTuple {
                (
                    [&]() -> typename Typelist::TypeAtIndex_MF<NodeValueTypes, Is> {
                        using Field = typename Typelist::TypeAtIndex_MF<Fields, Is>::Type;
                        if constexpr (Typelist::Contains_MF<ConstructibleFields, Field>::value) {
                            constexpr std::size_t index =
                                Typelist::FindType_MF<ConstructibleFields, Field>::value;
                            return std::get<index>(m_constructedValues);
                        }
                        else if constexpr (Typelist::Contains_MF<DeserializableValueTypes,
                                                                 Field>::value) {
                            constexpr std::size_t index =
                                Typelist::FindType_MF<DeserializableFields, Field>::value;
                            return std::get<index>(deserializedValues);
                        }
                        else {
                            static_assert(
                                AlwaysFalse_MF<Field>::value,
                                "Field not contained in Constructibles nor Deserializables");
                        }
                    }(),
                    ...);
            }(std::make_index_sequence<std::tuple_size_v<NodeValuesTuple>>{});
        }
    }

private:
    DeserializablesTuple getDeserializableValues(void) const
        requires(c_is_partially_deserializable)
    {
        assert(m_layout->getSegmentId().has_value() && "Layout must have a segmentId");
        using DataViews =
            typename detail::GetDeserializableDataViewTypes_MF<DeserializableFields>::Type;
        static_assert(Typelist::Size_MF<DataViews>::value == c_deserializable_fields_count,
                      "Typelists must have matching sizes.");

        const memory::id_t segmentId = m_layout->getSegmentId().value();
        // TODO make this more efficient than reading the entire segment when its possibly not even
        // used
        const std::vector<std::byte> serializedSegment = m_shmImpl.readSegment(segmentId);
        const std::span segmentView{serializedSegment};
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) -> DeserializablesTuple {
            (
                [&]() -> typename Typelist::TypeAtIndex_MF<DeserializableValueTypes, Is> {
                    using Field = typename Typelist::TypeAtIndex_MF<DeserializableFields, Is>::Type;
                    static_assert(Field::isFullyInlined, "Field must be fully inlined");
                    using DataView = typename Typelist::TypeAtIndex_MF<DataViews, Is>::Type;
                    using DeserializableType =
                        typename Typelist::TypeAtIndex_MF<DeserializableValueTypes, Is>::Type;
                    const auto& node = m_layout->get(std::type_identity<Field>{});
                    // TODO Implement
                    if constexpr (serialization::detail::MultiField<Field>) {
                        return DeserializableType{};
                    }
                    else if constexpr (serialization::detail::Field<Field>) {
                        return DeserializableType{};
                    }
                    else {
                        static_assert(AlwaysFalse_MF<Field>::value, "Unsupported FieldType.");
                    }
                }(),
                ...);
        }(std::make_index_sequence<c_deserializable_fields_count>{});
    }

    const serialization::Layout<S>* m_layout;
    const Shm& m_shmImpl;
    ConstructiblesTuple m_constructedValues;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SHM_READ_CONTEXT_H_
