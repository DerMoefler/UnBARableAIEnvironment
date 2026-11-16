#ifndef GRPC_CONNECTION_TPP_
#define GRPC_CONNECTION_TPP_

#include <type_traits>
#include "serialization/serialize_information.h"
#include "UnBARableAI/grpc_connection.hpp"

namespace UnBARableAINS {

namespace serialization {

template <>
struct SerializeInformation<GrpcConnection> {
    using Type = GrpcConnection;

    static constexpr bool c_serializable = true;

    struct F_Port : detail::Field_t {
        using Type = std::uint16_t;
    };

    using Fields = detail::Fields<F_Port>;

    static constexpr auto get(std::type_identity<F_Port>, const Type& value) noexcept ->
        typename F_Port::Type {
        return value.port;
    }

    static constexpr Type constructFromFields(const std::uint16_t port) noexcept {
        return Type{port};
    }
};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // GRPC_CONNECTION_TPP_
