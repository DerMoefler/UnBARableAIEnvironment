#ifndef SHM_READ_CONTEXT_H_
#define SHM_READ_CONTEXT_H_

#include <cstddef>
#include <vector>

#include "memory/shared_memory_impl.h"
#include "serialization/serialize_information.h"
#include "serialization/layout.h"
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

    using NodeValueTypes =
        typename serialization::detail::GetFieldsValueTypelist_MF<typename SI::Fields>::Type;

    using NodeValuesTuple = Typelist::Rebind_MF<std::tuple, NodeValueTypes>::Type;

    using DeserializableDataViewTypes =
        typename detail::GetDeserializableDataViewTypes_MF<typename SI::Fields>::Type;

public:
    ShmReadContext(const serialization::Layout<S>* layout, const Shm& shmImpl)
        : m_layout(layout)
        , m_shmImpl(shmImpl) {
        assert(m_layout);
        assert(m_layout->getSegmentId().has_value());

        const std::vector<std::byte> serializedSegment =
            m_shmImpl.readSegment(m_layout->getSegmentId().value());
    }

private:
    const serialization::Layout<S>* m_layout;
    const Shm& m_shmImpl;
    NodeValuesTuple m_nodeValues;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SHM_READ_CONTEXT_H_
