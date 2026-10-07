#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../include/UnBARableAI/unit_data.h"
#include "../../include/UnBARableAI/action.h"
#include "../../include/UnBARableAI/engine_status.h"

#include "../../include/UnBARableAI/bar_shared_memory.h"

namespace py = pybind11;

using UnitData = UnBARableAINS::unit::UnitData;
using Action = UnBARableAINS::Action;
using ActionId = UnBARableAINS::ActionId;
using EngineStatus = UnBARableAINS::EngineStatus;
using BarSharedMemory = UnBARableAINS::memory::BarSharedMemory;

PYBIND11_MODULE(bar_ai, m) {
    m.doc() = "Python bindings for UnBARableAI";

    // -------------------------
    // ActionId
    // -------------------------

    py::enum_<ActionId>(m, "ActionId")
        .value("MoveRight", ActionId::MoveRight)
        .value("MoveLeft", ActionId::MoveLeft)
        .value("MoveUp", ActionId::MoveUp)
        .value("MoveDown", ActionId::MoveDown)
        .value("Attack", ActionId::Attack)
        .export_values();

    // -------------------------
    // EngineStatus
    // -------------------------

    py::enum_<EngineStatus>(m, "EngineStatus")
        .value("UNSPECIFIED_ERROR", EngineStatus::UNSPECIFIED_ERROR)
        .value("GAME_ENDED", EngineStatus::GAME_ENDED)
        .value("TEAM_DIED", EngineStatus::TEAM_DIED)
        .value("AI_KILLED", EngineStatus::AI_KILLED)
        .value("AI_CRASHED", EngineStatus::AI_CRASHED)
        .value("AI_FAILED_TO_INIT", EngineStatus::AI_FAILED_TO_INIT)
        .value("CONNECTION_LOST", EngineStatus::CONNECTION_LOST)
        .value("OTHER_REASON_ERROR", EngineStatus::OTHER_REASON_ERROR)
        .value("RUNNING", EngineStatus::RUNNING)
        .export_values();

    // -------------------------
    // UnitData
    // -------------------------

    py::class_<UnitData>(m, "UnitData")
        .def(py::init<>())

        .def_readwrite("unit_id", &UnitData::unit_id)
        .def_readwrite("unit_def_id", &UnitData::unit_def_id)

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

    // -------------------------
    // SharedMemory
    // -------------------------

    py::class_<BarSharedMemory>(m, "SharedMemory")
        .def_static("create", &BarSharedMemory::create, py::arg("name"))

        .def_static("open", &BarSharedMemory::open, py::arg("name"))

        .def_static("remove", &BarSharedMemory::remove, py::arg("name"))

        .def("write_action", &BarSharedMemory::writeAction, py::arg("action"))

        .def("write_unit_data", &BarSharedMemory::writeUnitData, py::arg("unit_data"))

        .def("write_engine_status", &BarSharedMemory::writeEngineStatus, py::arg("status"))

        .def("read_all_units",
             [](BarSharedMemory& sharedMemory) { return sharedMemory.readAll<UnitData>(); })

        .def("read_all_engine_statuses",
             [](BarSharedMemory& sharedMemory) { return sharedMemory.readAll<EngineStatus>(); })

        .def("get_own_team_id", &BarSharedMemory::getOwnTeamId);
}
