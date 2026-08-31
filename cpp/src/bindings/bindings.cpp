#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../include/UnBARableAI/unit_data.h"
#include "../../include/UnBARableAI/action.h"

namespace py = pybind11;

PYBIND11_MODULE(bar_ai, m) {
    m.doc() = "Python bindings for UnBARableAI";

    using UnBARableAINS::unit::UnitData;
    using UnBARableAINS::Action;

    // -------------------------
    // UnitData
    // -------------------------
    py::class_<UnitData>(m, "UnitData")
        .def(py::init<>())

        .def_readwrite("unit_id", &UnitData::unit_id)
        .def_readwrite("unit_def_id", &UnitData::unit_def_id)
        .def_readwrite("unit_def_name", &UnitData::unit_def_name)
        .def_readwrite("human_name", &UnitData::human_name)

        .def_readwrite("team_id", &UnitData::team_id)
        .def_readwrite("ally_team_id", &UnitData::ally_team_id)

        .def_readwrite("health", &UnitData::health)
        .def_readwrite("max_health", &UnitData::max_health)

        .def_readwrite("pos_x", &UnitData::pos_x)
        .def_readwrite("pos_y", &UnitData::pos_y)
        .def_readwrite("pos_z", &UnitData::pos_z)

        .def_readwrite("los_radius", &UnitData::los_radius)
        .def_readwrite("air_los_radius", &UnitData::air_los_radius)

        .def_readwrite("is_dead", &UnitData::is_dead)
        .def_readwrite("being_built", &UnitData::being_built)

        .def_readwrite("build_progress", &UnitData::build_progress)
        .def_readwrite("capture_progress", &UnitData::capture_progress)
        .def_readwrite("paralyze_damage", &UnitData::paralyze_damage);

    // -------------------------
    // Action
    // -------------------------
    py::class_<Action>(m, "Action")
        .def(py::init<>())

        .def_readwrite("unit_id", &Action::unit_id)
        .def_readwrite("team_id", &Action::team_id)
        .def_readwrite("ally_team_id", &Action::ally_team_id)
        .def_readwrite("action_id", &Action::action_id)
        .def_readwrite("target_unit_id", &Action::target_unit_id);
}