/*!
 * \struct UnitData
 *
 * \copyright PHWT
 * \author Team UnBARableAI
 *
 * \brief Contains all relevant runtime information about a unit.
 *
 * This structure stores the current state of a game unit. it can be stored in the shared memory. All units in the game have forms the observation.
 */
#pragma once
namespace UnBARableAINS {

namespace unit {

struct UnitData {
    int unit_id;
    int unit_def_id;
    std::string unit_def_name;
    std::string human_name;
    int team_id;
    int ally_team_id;
    float health;
    float max_health;
    float pos_x;
    float pos_y;
    float pos_z;
    float los_radius;
    float air_los_radius;
    bool is_dead;
    bool being_built;
    float build_progress;
    float capture_progress;
    float paralyze_damage;
};

}; // namespace unit

}; // namespace UnBARableAI