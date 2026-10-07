#ifndef ENGINE_STATUS_TPP_
#define ENGINE_STATUS_TPP_

#include <cstdint>
#include <type_traits>

#include "serialization/serialize_information.h"
#include "UnBARableAI/engine_status.h"

namespace UnBARableAINS::serialization {

template <>
struct SerializeInformation<EngineStatus> {
    using Type = EngineStatus;
    using UnderlyingType = std::underlying_type_t<Type>;

    inline static constexpr bool c_serializable = true;
    inline static constexpr bool c_inline = true;

    struct F_Value : detail::Field_t {
        using Type = UnderlyingType;
    };

    using Fields = detail::Fields<F_Value>;

    inline static constexpr UnderlyingType get(
        std::type_identity<F_Value>,
        const Type status
    ) noexcept {
        return static_cast<UnderlyingType>(status);
    }

    inline static constexpr Type constructFromFields(
        const UnderlyingType value
    ) noexcept {
        return static_cast<Type>(value);
    }
};

static_assert(
    Serializable<EngineStatus>,
    "EngineStatus is not serializable!"
);

static_assert(
    detail::ConstructibleFromFields<EngineStatus>,
    "EngineStatus cannot be constructed from fields!"
);

}  // namespace UnBARableAINS::serialization

#endif  // ENGINE_STATUS_TPP_