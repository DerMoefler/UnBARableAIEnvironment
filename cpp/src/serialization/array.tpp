#ifndef ARRAY_TPP_
#define ARRAY_TPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <utility>

#include "serialization/serialize_information.h"
#include "serialization/trivial_serializable.hpp"

namespace UnBARableAINS {

namespace serialization {

template <TrivialSerializable T, std::size_t N>
struct SerializeInformation<std::array<T, N>> {
    using Type = std::array<T, N>;
    using TypeSI = SerializeInformation<T>;

    using Fields = detail::Fields<>;

    static constexpr bool c_serializable = true;

    static constexpr std::size_t c_element_size = getSerializableSize(std::type_identity<T>{});

    static constexpr std::size_t c_serialized_size = c_element_size * N;

    static constexpr std::array<std::byte, c_serialized_size> serialize(const Type& value) {
        return serializeImpl(value, std::make_index_sequence<N>{});
    }

    static constexpr Type deserialize(
        const std::span<const std::byte, c_serialized_size> serialized) {
        return deserializeImpl(serialized, std::make_index_sequence<N>{});
    }

private:
    template <std::size_t... Is>
    static constexpr std::array<std::byte, c_serialized_size> serializeImpl(
        const Type& value, std::index_sequence<Is...>) {
        std::array<std::byte, c_serialized_size> result{};
        (std::ranges::copy(serialization::serialize(value[Is]),
                           result.begin() + Is * c_element_size),
         ...);
        return result;
    }

    template <std::size_t... Is>
    static constexpr Type deserializeImpl(
        const std::span<const std::byte, c_serialized_size> serialized,
        std::index_sequence<Is...>) {
        return {TypeSI::deserialize(
            serialized.template subspan<Is * c_element_size, c_element_size>())...};
    }
};

namespace test {

static_assert(TrivialSerializable<std::array<std::size_t, 10>>,
              "std::array is not trivially serializable");

constexpr std::array<std::size_t, 5> arrayData = {0, 1, 200, std::numeric_limits<int>::max(),
                                                  std::numeric_limits<std::size_t>::max()};

constexpr auto serializedArrayData =
    SerializeInformation<std::array<std::size_t, 5>>::serialize(arrayData);
constexpr auto deserializedArrayData =
    SerializeInformation<std::array<std::size_t, 5>>::deserialize(serializedArrayData);

static_assert(arrayData == deserializedArrayData, "std::array de-/serialization does not work");

}  // namespace test

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
