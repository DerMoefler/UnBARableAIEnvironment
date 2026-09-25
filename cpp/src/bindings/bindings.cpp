#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../include/UnBARableAI/unit_data.h"
#include "../../include/UnBARableAI/action.h"

#include "../memory/shared_memory.h"
#include "../memory/shared_memory_posix.h"

namespace py = pybind11;

using UnitData = UnBARableAINS::unit::UnitData;
using Action = UnBARableAINS::Action;
using ActionId = UnBARableAINS::ActionId;

using BarSharedMemory =
    UnBARableAINS::memory::SharedMemory<
        UnBARableAINS::memory::SharedMemoryPosix,
        UnitData,
        Action>;

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

    // -------------------------
    // SharedMemory
    // -------------------------
    py::class_<BarSharedMemory>(m, "SharedMemory")
        .def_static(
            "create",
            const std::string& name {
                return BarSharedMemory::create(name);
            },
            py::arg("name"))

        .def_static(
            "open",
            const std::string& name {
                return BarSharedMemory::open(name);
            },
            py::arg("name"))

        .def_static(
            "remove",
            const std::string& name {
                BarSharedMemory::remove(name);
            },
            py::arg("name"))

        .def(
            "write_action",
            BarSharedMemory& sharedMemory, const Action& action {
                return sharedMemory.write(action);
            },
            py::arg("action"))

        .def(
            "write_unit_data",
            BarSharedMemory& sharedMemory, const UnitData& unitData {
                return sharedMemory.write(unitData);
            },
            py::arg("unit_data"));
}