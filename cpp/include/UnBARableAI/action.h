#pragma once

#include <cstdint>

namespace UnBARableAINS {

enum class ActionId : uint32_t {
    MoveRight = 1,
    MoveLeft = 2,
    MoveUp = 3,
    MoveDown = 4,
    Attack = 5
};

struct Action {
    bool operator==(const Action& other) const = default;

    uint32_t unit_id;
    uint32_t team_id;
    uint32_t ally_team_id;
    ActionId action_id;
    uint32_t target_unit_id;  // Only used for Attack
};

}  // namespace UnBARableAINS
