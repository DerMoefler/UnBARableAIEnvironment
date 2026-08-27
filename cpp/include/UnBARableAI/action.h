#pragma once

namespace UnBARableAINS {

struct Action {
    int unit_id;
    int unit_def_id;
    std::string unit_def_name;
    std::string human_name;
    int team_id;
    int ally_team_id;
    int action_id; // 1: move right, 2: move left, 3: move up, 4: move down, 5: attack
    int target_unit_id; // Only used for attack action
};

}; // namespace UnBARableAI