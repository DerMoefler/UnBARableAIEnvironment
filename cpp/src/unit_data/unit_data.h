#pragma once
namespace UnBARableAI {

namespace unit {

struct UnitData {
    float health;
    int allyTeam;
    float3 position;
    std::vector<Unit::Command> currentCommands;
};

}; // namespace unit

}; // namespace UnBARableAI