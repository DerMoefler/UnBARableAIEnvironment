#include <array>
#include <cstddef>
#include <span>
#include <type_traits>

#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

template <typename T>
    requires std::is_enum_v<T>
struct SerializeInformation<T> {
    using Type = T;
    using UnderlyingType = std::underlying_type_t<T>;
    using Fields = detail::Fields<>;

    inline static constexpr bool c_serializable =
        SerializeInformation<UnderlyingType>::c_serializable;
    inline static constexpr std::size_t c_serialized_size =
        SerializeInformation<UnderlyingType>::c_serialized_size;

    inline static constexpr std::array<std::byte, sizeof(UnderlyingType)> serialize(
        const Type value) {
        return SerializeInformation<UnderlyingType>::serialize(static_cast<UnderlyingType>(value));
    }

    inline static constexpr Type deserialize(
        const std::span<const std::byte, sizeof(UnderlyingType)> serialized) {
        return static_cast<Type>(SerializeInformation<UnderlyingType>::deserialize(serialized));
    }
};

namespace test {
enum class TestAction : std::size_t { ValueA = 1, ValueB = 3, ValueC = 100, ValueD = 101 };

static_assert(Serializable<TestAction>, "TestAction enum does not satisfy Serializable.");
}  // namespace test

}  // namespace serialization

}  // namespace UnBARableAINS
