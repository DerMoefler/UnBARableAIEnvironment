#ifndef ENGINE_INFORMATION_TPP_
#define ENGINE_INFORMATION_TPP_

#include <type_traits>
#include "serialization/serialize_information.h"
#include "UnBARableAI/engine_information.hpp"
#include "UnBARableAI/engine_status.h"
#include "UnBARableAI/grpc_connection.hpp"

#include "serialization/engine_status.tpp"
#include "serialization/grpc_connection.tpp"

namespace UnBARableAINS::serialization {

template <>
struct SerializeInformation<EngineInformation> {
    using Type = EngineInformation;

    static constexpr bool c_serializable =
        Serializable<EngineStatus> && Serializable<GrpcConnection>;
    static constexpr bool c_inline = true;

    struct F_EngineStatus : detail::Field_t {
        using Type = EngineStatus;
    };

    struct F_GrpcConnection : detail::Field_t {
        using Type = GrpcConnection;
    };

    struct F_IsTruncated : detail::Field_t {
        using Type = bool;
    };

    using Fields = detail::Fields<F_EngineStatus, F_GrpcConnection, F_IsTruncated>;

    static constexpr EngineStatus get(std::type_identity<F_EngineStatus>,
                                      const Type& info) noexcept {
        return info.engineStatus;
    }

    static constexpr GrpcConnection get(std::type_identity<F_GrpcConnection>,
                                        const Type& info) noexcept {
        return info.grpcConnection;
    }

    static constexpr bool get(std::type_identity<F_IsTruncated>, const Type& info) noexcept {
        return info.isTruncated;
    }

    static constexpr Type constructFromFields(const EngineStatus status,
                                              const GrpcConnection connection,
                                              const bool isTruncated) {
        return Type{
            .engineStatus = status, .grpcConnection = connection, .isTruncated = isTruncated};
    }
};

}  // namespace UnBARableAINS::serialization

#endif  // ENGINE_INFORMATION_TPP_
