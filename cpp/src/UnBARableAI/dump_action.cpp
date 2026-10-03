#include "UnBARableAI/debug/dump_action.hpp"

#include <magic_enum/magic_enum.hpp>

#include "utility/debug/print_helpers.hpp"

namespace UnBARableAINS {

namespace debug {

void dumpAction(std::ostream& out, const Action& action, std::size_t indentationLevel) {
    makeIndentation(out, indentationLevel);
    out << "Unit ID: " << action.unit_id << '\n';
    makeIndentation(out, indentationLevel);
    out << "Team ID: " << action.team_id << '\n';
    makeIndentation(out, indentationLevel);
    out << "AllyTeam ID: " << action.ally_team_id << '\n';
    makeIndentation(out, indentationLevel);
    out << "Action ID: " << magic_enum::enum_name(action.action_id) << '\n';
    makeIndentation(out, indentationLevel);
    out << "Target Unit ID: " << action.target_unit_id;
    out << std::endl;
}

}  // namespace debug

}  // namespace UnBARableAINS
