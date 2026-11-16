#ifndef GRPC_CONNECTION_H_
#define GRPC_CONNECTION_H_

#include <cstdint>

namespace UnBARableAINS {

/**
 * \brief Simple struct for holding information about a gRPC connection.
 *
 * Currently only holds port but can easily be extended. This class is merely meant to be serialized
 * into BarSharedMemory.
 */
struct GrpcConnection {
    /// \brief Port of the gRPC server.
    std::uint16_t port;
};

}  // namespace UnBARableAINS

#endif  // GRPC_CONNECTION_H_
