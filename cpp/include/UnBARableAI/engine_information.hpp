#ifndef ENGINE_INFORMATION_H_
#define ENGINE_INFORMATION_H_

#include "UnBARableAI/engine_status.h"
#include "UnBARableAI/grpc_connection.hpp"

namespace UnBARableAINS {

/**
 * \brief Contains information relevant for the engine.
 *
 * Meant to be cheap to write (shouldnt grow to like a few megabyte lol) or have the user write
 * explicit serialization functionality using getFieldOffset and serializing the field manually.
 */
struct EngineInformation {
    bool operator==(const EngineInformation& other) const = default;
    /// \brief \ref EngineStatus.
    EngineStatus engineStatus;
    /// \brief \ref GrpcConnection.
    GrpcConnection grpcConnection;
    /**
     * \brief Flag for the python side to communicate that the engine must reload.
     *
     * Flag should be set if the maximum number of steps is reached inside the BAR Environment,
     * meaning that the episode is truncated and the engine thus has to reload.
     * The engine is expected to check for this flag in the EVENT_UPDATE. If it is set to true, the
     * engine should terminate the game gracefully, meaning the AI should request the shutdown in a
     * way that it still receives an EVENT_RELEASED. It should then clear this flag in the following
     * episode's EVENT_INIT.
     */
    bool isTruncated;
};

}  // namespace UnBARableAINS

#endif  // ENGINE_INFORMATION_H_
