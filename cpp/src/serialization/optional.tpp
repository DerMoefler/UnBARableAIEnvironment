#ifndef OPTIONAL_TPP_
#define OPTIONAL_TPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

/**
 * \brief Implementation for a std::optional<T> where \p T satisfies \ref detail::ConstSize.
 * \tparam T A Serializable with constant size.
 *
 * This implementation uses a boolean flag to encode whether or not a value is present.
 * It occupies memory for the value nonetheless, since the value's size is known at compile time.
 * This avoids having to resize later on.
 *
 * The layout looks rougly like this:
\verbatim
[has_value][value]
\endverbatim
where value is all zeros if has_value = false.
 */
template <Serializable T>
    requires detail::ConstSize<T>
struct SerializeInformation<std::optional<T>> {
    using Type = std::optional<T>;
    using BoolSI = SerializeInformation<bool>;
    using TypeSI = SerializeInformation<T>;

    using Fields = detail::Fields<>;

    inline static constexpr bool c_serializable = Serializable<T>;
    inline static constexpr std::size_t c_serialized_size =
        BoolSI::c_serialized_size + TypeSI::c_serialized_size;

    static constexpr std::array<std::byte, c_serialized_size> serialize(const Type& value) {
        std::array<std::byte, c_serialized_size> result{};

        const bool hasValue = value.has_value();
        const std::array<std::byte, BoolSI::c_serialized_size> serializedHasValue =
            SerializeInformation<bool>::serialize(hasValue);
        std::ranges::copy(serializedHasValue, result.begin());

        if (hasValue) {
            const std::array<std::byte, TypeSI::c_serialized_size> serializedType =
                TypeSI::serialize(value.value());
            std::ranges::copy(serializedType, result.begin() + BoolSI::c_serialized_size);
        }

        return result;
    }

    static constexpr Type deserialize(
        const std::span<const std::byte, c_serialized_size> serialized) {
        const auto hasValueView = serialized.template first<BoolSI::c_serialized_size>();
        const bool hasValue = SerializeInformation<bool>::deserialize(hasValueView);

        if (!hasValue) {
            return std::nullopt;
        }

        const auto valueView =
            serialized.template subspan<BoolSI::c_serialized_size, TypeSI::c_serialized_size>();
        return TypeSI::deserialize(valueView);
    }
};

namespace test {

constexpr std::size_t value{0xFF};
constexpr std::optional<std::size_t> optional{value};
using SI = SerializeInformation<std::optional<std::size_t>>;

constexpr auto serializedOptional = SI::serialize(optional);
constexpr auto deserializedOptional = SI::deserialize(serializedOptional);
static_assert(deserializedOptional.has_value(), "Deserialized object must have a value");
static_assert(deserializedOptional.value() == value, "Deserialized values must match");

constexpr auto nulloptSerialized = SI::serialize(std::nullopt);
constexpr auto nulloptDeserialized = SI::deserialize(nulloptSerialized);
static_assert(!nulloptDeserialized.has_value(), "std::nullopt deserialized must have no value");
static_assert(nulloptDeserialized == std::nullopt, "std::nullopt incorecctly deserialized");

}  // namespace test

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
