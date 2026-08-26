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