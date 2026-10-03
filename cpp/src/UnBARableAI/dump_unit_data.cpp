#include "UnBARableAI/debug/dump_unit_data.hpp"

#include "utility/debug/print_helpers.hpp"

namespace UnBARableAINS {

namespace debug {

void dumpUnitData(std::ostream& out, const unit::UnitData& unitData, std::size_t indentationLevel) {
    makeIndentation(out, indentationLevel);
    out << "Unit ID: " << unitData.unit_id << '\n';
    makeIndentation(out, indentationLevel);
    out << "UnitDef ID: " << unitData.unit_def_id << '\n';
    makeIndentation(out, indentationLevel);
    out << "Team ID: " << unitData.team_id << '\n';
    makeIndentation(out, indentationLevel);
    out << "AllyTeam ID: " << unitData.ally_team_id << '\n';
    makeIndentation(out, indentationLevel);
    out << "Health: " << unitData.health << '\n';
    makeIndentation(out, indentationLevel);
    out << "Max Health: " << unitData.max_health << '\n';
    makeIndentation(out, indentationLevel);
    out << "Position: (" << unitData.pos_x << ", " << unitData.pos_y << ", " << unitData.pos_z
        << ")\n";
    makeIndentation(out, indentationLevel);
    out << "LOS Radius: " << unitData.los_radius << '\n';
    makeIndentation(out, indentationLevel);
    out << "Air LOS Radius: " << unitData.air_los_radius << '\n';
    makeIndentation(out, indentationLevel);
    out << "Is Dead: " << getBoolValueDisplayName(unitData.is_dead) << '\n';
    makeIndentation(out, indentationLevel);
    out << "Being built: " << getBoolValueDisplayName(unitData.being_built) << '\n';
    makeIndentation(out, indentationLevel);
    out << "Build Progress: " << unitData.build_progress << '\n';
    makeIndentation(out, indentationLevel);
    out << "Capture Progress: " << unitData.capture_progress << '\n';
    makeIndentation(out, indentationLevel);
    out << "Paralyze Damage: " << unitData.paralyze_damage;
    out << std::endl;
}

}  // namespace debug

}  // namespace UnBARableAINS
