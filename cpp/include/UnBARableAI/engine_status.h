#pragma once

#include <cstdint>

namespace UnBARableAINS {

enum class EngineStatus : std::uint8_t {
    UNSPECIFIED_ERROR = 0,
    GAME_ENDED = 1,
    TEAM_DIED = 2,
    AI_KILLED = 3,
    AI_CRASHED = 4,
    AI_FAILED_TO_INIT = 5,
    CONNECTION_LOST = 6,
    OTHER_REASON_ERROR = 7,
    RUNNING = 8
};

}  // namespace UnBARableAINS