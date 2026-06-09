#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../unit/unit.h"
#include "../unit/pawn.h"
#include "../unit_data/unit_data.h"

namespace py = pybind11;

PYBIND11_MODULE(bar_ai, m) {
    m.doc() = "Python bindings for UnBARableAI";

    using UnBARableAI::unit::Unit;
    using UnBARableAI::unit::Pawn;
    using UnBARableAI::unit::UnitData;

    // -------------------------
    // UnitData
    // -------------------------
    // UnitData ist ein einfacher Struct-Typ mit öffentlichen Feldern.
    // Solche Typen bindet man direkt mit def_readwrite. [1](https://pybind11.readthedocs.io/en/stable/classes.html)
    py::class_<UnitData>(m, "UnitData")
        .def(py::init<>())
        .def_readwrite("health", &UnitData::health)
        .def_readwrite("team", &UnitData::team)
        .def_readwrite("xPosition", &UnitData::xPosition)
        .def_readwrite("yPosition", &UnitData::yPosition)
        .def_readwrite("zPosition", &UnitData::zPosition)
        .def_readwrite("hasCurrentCommand", &UnitData::hasCurrentCommand)
        .def_readwrite("unitID", &UnitData::unitID)
        .def_readwrite("unitType", &UnitData::unitType);

    // -------------------------
    // Unit (abstrakte Basisklasse)
    // -------------------------
    // Keine .def(py::init<>()) !
    // Unit ist abstrakt und nicht direkt konstruierbar. [2](https://pybind11.readthedocs.io/en/stable/advanced/classes.html)
    py::class_<Unit>(m, "Unit")
        .def("getHealth", &Unit::getHealth)
        .def("getTeam", &Unit::getTeam)
        .def("getXPosition", &Unit::getXPosition)
        .def("getYPosition", &Unit::getYPosition)
        .def("getZPosition", &Unit::getZPosition)
        .def("hasCurrentCommand", &Unit::hasCurrentCommand)
        .def("getUnitID", &Unit::getUnitID)
        .def("getUnitType", &Unit::getUnitType)
        .def("getUnitsInSight", &Unit::getUnitsInSight)
        .def("getEnemyUnitsInSight", &Unit::getEnemyUnitsInSight);

    // -------------------------
    // Pawn : Unit
    // -------------------------
    // Vererbung wird mit py::class_<Pawn, Unit> angegeben. [1](https://pybind11.readthedocs.io/en/stable/classes.html)[2](https://pybind11.readthedocs.io/en/stable/advanced/classes.html)
    py::class_<Pawn, Unit>(m, "Pawn")
        .def(py::init<UnitData>(), py::arg("data"))
        .def("getHealth", &Pawn::getHealth)
        .def("getTeam", &Pawn::getTeam)
        .def("getXPosition", &Pawn::getXPosition)
        .def("getYPosition", &Pawn::getYPosition)
        .def("getZPosition", &Pawn::getZPosition)
        .def("hasCurrentCommand", &Pawn::hasCurrentCommand)
        .def("getUnitID", &Pawn::getUnitID)
        .def("getUnitType", &Pawn::getUnitType)
        .def("getUnitsInSight", &Pawn::getUnitsInSight)
        .def("getEnemyUnitsInSight", &Pawn::getEnemyUnitsInSight);
}