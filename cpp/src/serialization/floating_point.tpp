#include <array>
#include <concepts>
#include <cstddef>
#include <span>

#include "serialization/serialize_information.h"
#include "utility/unsigned_of_size.hpp"

namespace UnBARableAINS {

namespace serialization {

template <std::floating_point T>
struct SerializeInformation<T> {
    using Type = T;
    using Fields = detail::Fields<>;

    using UnsignedType = util::UnsignedOfSize_T<sizeof(T)>;

    inline static constexpr bool c_serializable = true;
    inline static constexpr std::size_t c_serialized_size = sizeof(Type);

    inline static constexpr std::array<std::byte, sizeof(Type)> serialize(const Type value) {
        return SerializeInformation<UnsignedType>::serialize(std::bit_cast<UnsignedType>(value));
    }

    inline static constexpr Type deserialize(
        const std::span<const std::byte, sizeof(Type)> serialized) {
        return std::bit_cast<Type>(SerializeInformation<UnsignedType>::deserialize(serialized));
    }
};

}  // namespace serialization

}  // namespace UnBARableAINS
