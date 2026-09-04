#pragma once

namespace UnBARableAINS {

enum class ActionId : int {
    MoveRight = 1,
    MoveLeft  = 2,
    MoveUp    = 3,
    MoveDown  = 4,
    Attack    = 5
};

struct Action {
    int unit_id;
    int team_id;
    int ally_team_id;
    ActionId action_id;
    int target_unit_id; // Only used for Attack
};

} // namespace UnBARableAINS