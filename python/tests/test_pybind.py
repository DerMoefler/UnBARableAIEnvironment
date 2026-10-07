# test_bar_ai.py

import os
import time

import pytest

import bar_ai


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

REQUIRED_EXPORTS = (
    "UnitData",
    "Action",
    "ActionId",
    "EngineStatus",
    "SharedMemory",
)

REQUIRED_SHARED_MEMORY_METHODS = (
    "create",
    "open",
    "remove",
    "write_unit_data",
    "write_action",
    "write_engine_status",
    "read_all_units",
    "read_all_engine_statuses",
    "get_own_team_id",
)

ACTION_ID_NAMES = (
    "MoveRight",
    "MoveLeft",
    "MoveUp",
    "MoveDown",
    "Attack",
)

ENGINE_STATUS_NAMES = (
    "UNSPECIFIED_ERROR",
    "GAME_ENDED",
    "TEAM_DIED",
    "AI_KILLED",
    "AI_CRASHED",
    "AI_FAILED_TO_INIT",
    "CONNECTION_LOST",
    "OTHER_REASON_ERROR",
    "RUNNING",
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def create_unit(
    unit_id,
    team_id,
    ally_team_id,
    unit_def_id,
    pos_x,
    pos_y,
    pos_z,
    health=100.0,
    max_health=100.0,
):
    """Erzeugt eine vollständig initialisierte UnitData-Instanz."""
    unit = bar_ai.UnitData()

    unit.unit_id = unit_id
    unit.unit_def_id = unit_def_id

    unit.team_id = team_id
    unit.ally_team_id = ally_team_id

    unit.health = health
    unit.max_health = max_health

    unit.pos_x = pos_x
    unit.pos_y = pos_y
    unit.pos_z = pos_z

    unit.los_radius = 500.0
    unit.air_los_radius = 350.0

    unit.is_dead = False
    unit.being_built = False

    unit.build_progress = 1.0
    unit.capture_progress = 0.0
    unit.paralyze_damage = 0.0

    return unit


def assert_fields(instance, expected_values, object_name):
    """Vergleicht Attribute eines gebundenen C++-Objekts."""
    for field_name, expected_value in expected_values.items():
        actual_value = getattr(instance, field_name)

        assert actual_value == expected_value, (
            f"{object_name}.{field_name}: "
            f"Erwartet {expected_value!r}, "
            f"erhalten {actual_value!r}"
        )


def unit_ids(units):
    """Extrahiert Unit-IDs aus einer UnitData-Liste."""
    return [unit.unit_id for unit in units]


# ---------------------------------------------------------------------------
# Fixtures für Testobjekte
# ---------------------------------------------------------------------------

@pytest.fixture
def friendly_agent():
    return create_unit(
        unit_id=42,
        team_id=1,
        ally_team_id=0,
        unit_def_id=101,
        pos_x=100.0,
        pos_y=20.0,
        pos_z=100.0,
        health=3000.0,
        max_health=3000.0,
    )


@pytest.fixture
def friendly_unit():
    return create_unit(
        unit_id=43,
        team_id=1,
        ally_team_id=0,
        unit_def_id=3,
        pos_x=150.0,
        pos_y=20.0,
        pos_z=120.0,
        health=100.0,
        max_health=250.0,
    )


@pytest.fixture
def visible_enemy_one():
    return create_unit(
        unit_id=99,
        team_id=2,
        ally_team_id=1,
        unit_def_id=201,
        pos_x=800.0,
        pos_y=20.0,
        pos_z=700.0,
        health=2800.0,
        max_health=3000.0,
    )


@pytest.fixture
def visible_enemy_two():
    return create_unit(
        unit_id=100,
        team_id=3,
        ally_team_id=1,
        unit_def_id=202,
        pos_x=900.0,
        pos_y=20.0,
        pos_z=750.0,
        health=75.0,
        max_health=100.0,
    )


@pytest.fixture
def action():
    result = bar_ai.Action()

    result.unit_id = 42
    result.team_id = 1
    result.ally_team_id = 0
    result.action_id = bar_ai.ActionId.Attack
    result.target_unit_id = 99

    return result


@pytest.fixture
def test_units(
    friendly_agent,
    friendly_unit,
    visible_enemy_one,
    visible_enemy_two,
):
    return [
        friendly_agent,
        friendly_unit,
        visible_enemy_one,
        visible_enemy_two,
    ]


# ---------------------------------------------------------------------------
# Shared-Memory-Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_shared_memory(test_units, action):
    """
    Erstellt ein Shared Memory und befüllt es in dieser Reihenfolge:

    0. Friendly Agent
    1. Friendly Unit
    2. Enemy 1
    3. Enemy 2
    4. Action
    5. EngineStatus.RUNNING

    Die erste Unit bestimmt die eigene Team-ID.
    """
    shared_memory_name = (
        f"/bar_ai_test_{os.getpid()}_{time.time_ns()}"
    )

    shared_memory = None

    # Eventuelle Altlast entfernen. Bei einem eindeutigen Namen sollte
    # normalerweise nichts vorhanden sein.
    try:
        bar_ai.SharedMemory.remove(shared_memory_name)
    except Exception:
        pass

    try:
        shared_memory = bar_ai.SharedMemory.create(
            shared_memory_name
        )

        serializable_ids = [
            shared_memory.write_unit_data(unit)
            for unit in test_units
        ]

        serializable_ids.append(
            shared_memory.write_action(action)
        )

        serializable_ids.append(
            shared_memory.write_engine_status(
                bar_ai.EngineStatus.RUNNING
            )
        )

        yield {
            "name": shared_memory_name,
            "memory": shared_memory,
            "serializable_ids": serializable_ids,
        }

    finally:
        # Referenz auf das gebundene C++-/mmap-Objekt freigeben,
        # bevor das Shared Memory entfernt wird.
        shared_memory = None

        try:
            bar_ai.SharedMemory.remove(shared_memory_name)
        except Exception as cleanup_error:
            pytest.fail(
                "Shared Memory konnte nicht entfernt werden: "
                f"{type(cleanup_error).__name__}: "
                f"{cleanup_error}"
            )


# ---------------------------------------------------------------------------
# Modul- und API-Tests
# ---------------------------------------------------------------------------

def test_bar_ai_module_can_be_imported():
    assert bar_ai is not None


@pytest.mark.parametrize("export_name", REQUIRED_EXPORTS)
def test_required_export_exists(export_name):
    assert hasattr(bar_ai, export_name), (
        f"Fehlende Klasse oder Enum: {export_name}"
    )


@pytest.mark.parametrize(
    "method_name",
    REQUIRED_SHARED_MEMORY_METHODS,
)
def test_required_shared_memory_method_exists(method_name):
    assert hasattr(bar_ai.SharedMemory, method_name), (
        "Fehlende SharedMemory-Methode: "
        f"{method_name}"
    )

    method = getattr(bar_ai.SharedMemory, method_name)

    assert callable(method), (
        f"SharedMemory.{method_name} ist nicht aufrufbar"
    )


# ---------------------------------------------------------------------------
# Enum-Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action_id_name", ACTION_ID_NAMES)
def test_action_id_member_exists(action_id_name):
    assert hasattr(bar_ai.ActionId, action_id_name), (
        f"Fehlender ActionId-Wert: {action_id_name}"
    )

    value = getattr(bar_ai.ActionId, action_id_name)

    assert isinstance(value, bar_ai.ActionId), (
        f"ActionId.{action_id_name} ist kein ActionId, "
        f"sondern {type(value).__name__}"
    )


@pytest.mark.parametrize(
    "engine_status_name",
    ENGINE_STATUS_NAMES,
)
def test_engine_status_member_exists(engine_status_name):
    assert hasattr(bar_ai.EngineStatus, engine_status_name), (
        f"Fehlender EngineStatus-Wert: {engine_status_name}"
    )

    value = getattr(
        bar_ai.EngineStatus,
        engine_status_name,
    )

    assert isinstance(value, bar_ai.EngineStatus), (
        f"EngineStatus.{engine_status_name} ist kein "
        f"EngineStatus, sondern {type(value).__name__}"
    )


def test_engine_status_values_are_distinct():
    running = bar_ai.EngineStatus.RUNNING
    game_ended = bar_ai.EngineStatus.GAME_ENDED

    assert running != game_ended


# ---------------------------------------------------------------------------
# UnitData-Tests
# ---------------------------------------------------------------------------

def test_unit_data_can_be_created(friendly_agent):
    assert isinstance(friendly_agent, bar_ai.UnitData)


def test_friendly_agent_fields(friendly_agent):
    expected_values = {
        "unit_id": 42,
        "unit_def_id": 101,
        "team_id": 1,
        "ally_team_id": 0,
        "health": 3000.0,
        "max_health": 3000.0,
        "pos_x": 100.0,
        "pos_y": 20.0,
        "pos_z": 100.0,
        "los_radius": 500.0,
        "air_los_radius": 350.0,
        "is_dead": False,
        "being_built": False,
        "build_progress": 1.0,
        "capture_progress": 0.0,
        "paralyze_damage": 0.0,
    }

    assert_fields(
        friendly_agent,
        expected_values,
        "friendly_agent",
    )


# ---------------------------------------------------------------------------
# Action-Tests
# ---------------------------------------------------------------------------

def test_action_can_be_created(action):
    assert isinstance(action, bar_ai.Action)


def test_action_fields(action):
    expected_values = {
        "unit_id": 42,
        "team_id": 1,
        "ally_team_id": 0,
        "action_id": bar_ai.ActionId.Attack,
        "target_unit_id": 99,
    }

    assert_fields(
        action,
        expected_values,
        "action",
    )


# ---------------------------------------------------------------------------
# Shared-Memory-Schreibtests
# ---------------------------------------------------------------------------

def test_write_methods_return_integer_ids(
    populated_shared_memory,
):
    serializable_ids = (
        populated_shared_memory["serializable_ids"]
    )

    assert all(
        isinstance(serializable_id, int)
        for serializable_id in serializable_ids
    ), (
        "Alle write-Funktionen müssen Integer-IDs "
        f"zurückgeben: {serializable_ids!r}"
    )


def test_serializable_ids_are_non_negative(
    populated_shared_memory,
):
    serializable_ids = (
        populated_shared_memory["serializable_ids"]
    )

    assert all(
        serializable_id >= 0
        for serializable_id in serializable_ids
    ), (
        "Alle Serializable-IDs müssen nichtnegativ sein: "
        f"{serializable_ids!r}"
    )


def test_serializable_ids_are_unique(
    populated_shared_memory,
):
    serializable_ids = (
        populated_shared_memory["serializable_ids"]
    )

    assert len(set(serializable_ids)) == len(serializable_ids), (
        "Serializable-IDs sind nicht eindeutig: "
        f"{serializable_ids!r}"
    )


def test_serializable_ids_follow_write_order(
    populated_shared_memory,
):
    serializable_ids = (
        populated_shared_memory["serializable_ids"]
    )

    assert serializable_ids == [0, 1, 2, 3, 4, 5], (
        "Unerwartete Serializable-ID-Reihenfolge. "
        "Erwartet [0, 1, 2, 3, 4, 5], "
        f"erhalten {serializable_ids!r}"
    )


# ---------------------------------------------------------------------------
# Shared-Memory-Lesetests
# ---------------------------------------------------------------------------

def test_first_unit_determines_own_team_id(
    populated_shared_memory,
):
    shared_memory = populated_shared_memory["memory"]

    assert shared_memory.get_own_team_id() == 1


def test_read_all_units_returns_list(
    populated_shared_memory,
):
    shared_memory = populated_shared_memory["memory"]

    units = shared_memory.read_all_units()

    assert isinstance(units, list), (
        "read_all_units() sollte eine Python-Liste "
        f"zurückgeben, erhalten: {type(units).__name__}"
    )


def test_read_all_units_returns_only_units(
    populated_shared_memory,
):
    shared_memory = populated_shared_memory["memory"]

    units = shared_memory.read_all_units()

    assert len(units) == 4, (
        "read_all_units() sollte genau vier Units liefern. "
        "Action und EngineStatus müssen übersprungen werden. "
        f"Erhalten: {len(units)}"
    )

    assert all(
        isinstance(unit, bar_ai.UnitData)
        for unit in units
    )


def test_read_all_units_preserves_order(
    populated_shared_memory,
):
    shared_memory = populated_shared_memory["memory"]

    units = shared_memory.read_all_units()

    assert unit_ids(units) == [42, 43, 99, 100]


def test_read_all_units_preserves_team_ids(
    populated_shared_memory,
):
    shared_memory = populated_shared_memory["memory"]

    units = shared_memory.read_all_units()
    team_ids = [unit.team_id for unit in units]

    assert team_ids == [1, 1, 2, 3]


def test_read_all_units_preserves_data(
    populated_shared_memory,
):
    shared_memory = populated_shared_memory["memory"]

    units = shared_memory.read_all_units()

    expected_units = (
        {
            "unit_id": 42,
            "team_id": 1,
            "health": 3000.0,
        },
        {
            "unit_id": 43,
            "team_id": 1,
            "health": 100.0,
        },
        {
            "unit_id": 99,
            "team_id": 2,
            "health": 2800.0,
        },
        {
            "unit_id": 100,
            "team_id": 3,
            "health": 75.0,
        },
    )

    assert len(units) == len(expected_units)

    for index, expected_values in enumerate(expected_units):
        assert_fields(
            units[index],
            expected_values,
            f"units[{index}]",
        )


def test_units_can_be_divided_into_friendly_and_enemy(
    populated_shared_memory,
):
    shared_memory = populated_shared_memory["memory"]

    own_team_id = shared_memory.get_own_team_id()
    units = shared_memory.read_all_units()

    friendly_units = [
        unit
        for unit in units
        if unit.team_id == own_team_id
    ]

    enemy_units = [
        unit
        for unit in units
        if unit.team_id != own_team_id
    ]

    assert unit_ids(friendly_units) == [42, 43]
    assert unit_ids(enemy_units) == [99, 100]


# ---------------------------------------------------------------------------
# EngineStatus-Lesetests
# ---------------------------------------------------------------------------

def test_read_all_engine_statuses_returns_list(
    populated_shared_memory,
):
    shared_memory = populated_shared_memory["memory"]

    statuses = shared_memory.read_all_engine_statuses()

    assert isinstance(statuses, list), (
        "read_all_engine_statuses() sollte eine "
        "Python-Liste liefern, erhalten: "
        f"{type(statuses).__name__}"
    )


def test_read_all_engine_statuses_returns_only_statuses(
    populated_shared_memory,
):
    shared_memory = populated_shared_memory["memory"]

    statuses = shared_memory.read_all_engine_statuses()

    assert statuses == [bar_ai.EngineStatus.RUNNING]

    assert all(
        isinstance(status, bar_ai.EngineStatus)
        for status in statuses
    )


def test_read_methods_filter_serialized_types(
    populated_shared_memory,
):
    shared_memory = populated_shared_memory["memory"]

    units = shared_memory.read_all_units()
    statuses = shared_memory.read_all_engine_statuses()

    assert len(units) == 4, (
        "Action oder EngineStatus wurde fälschlich "
        "als UnitData gelesen"
    )

    assert len(statuses) == 1, (
        "UnitData oder Action wurde fälschlich "
        "als EngineStatus gelesen"
    )


# ---------------------------------------------------------------------------
# Shared Memory erneut öffnen
# ---------------------------------------------------------------------------

def test_existing_shared_memory_can_be_opened(
    populated_shared_memory,
):
    shared_memory_name = populated_shared_memory["name"]

    opened_shared_memory = bar_ai.SharedMemory.open(
        shared_memory_name
    )

    assert opened_shared_memory is not None


def test_opened_shared_memory_has_correct_team_id(
    populated_shared_memory,
):
    shared_memory_name = populated_shared_memory["name"]

    opened_shared_memory = bar_ai.SharedMemory.open(
        shared_memory_name
    )

    assert opened_shared_memory.get_own_team_id() == 1


def test_opened_shared_memory_contains_units(
    populated_shared_memory,
):
    shared_memory_name = populated_shared_memory["name"]

    opened_shared_memory = bar_ai.SharedMemory.open(
        shared_memory_name
    )

    reopened_units = opened_shared_memory.read_all_units()

    assert unit_ids(reopened_units) == [
        42,
        43,
        99,
        100,
    ]


def test_opened_shared_memory_contains_engine_status(
    populated_shared_memory,
):
    shared_memory_name = populated_shared_memory["name"]

    opened_shared_memory = bar_ai.SharedMemory.open(
        shared_memory_name
    )

    reopened_statuses = (
        opened_shared_memory.read_all_engine_statuses()
    )

    assert reopened_statuses == [
        bar_ai.EngineStatus.RUNNING
    ]