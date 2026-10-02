import os
import sys
import time


print("🚀 Starte bar_ai UnitData-, Action- und Shared-Memory-Test...")


# ----------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------

def fail(message, exception=None):
    """Beendet den Test mit einer verständlichen Fehlermeldung."""
    print(f"❌ {message}")

    if exception is not None:
        print(f"{type(exception).__name__}: {exception}")

    sys.exit(1)


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


def validate_fields(instance, expected_values, object_name):
    """Vergleicht Attribute eines gebundenen C++-Objekts."""
    for field_name, expected_value in expected_values.items():
        actual_value = getattr(instance, field_name)

        assert actual_value == expected_value, (
            f"{object_name}.{field_name}: "
            f"Erwartet {expected_value!r}, "
            f"erhalten {actual_value!r}"
        )

        print(
            f"✅ {object_name}.{field_name}: "
            f"{actual_value!r}"
        )


def unit_ids(units):
    """Extrahiert die BAR-Unit-IDs aus einer UnitData-Liste."""
    return [unit.unit_id for unit in units]


# ----------------------------------------------------
# Modul importieren
# ----------------------------------------------------

try:
    import bar_ai
    print("✅ import bar_ai erfolgreich")
except Exception as e:
    fail("import bar_ai fehlgeschlagen", e)

print("📦 Modul:", bar_ai)


# ----------------------------------------------------
# Exportierte Klassen prüfen
# ----------------------------------------------------

required_classes = (
    "UnitData",
    "Action",
    "ActionId",
    "SharedMemory",
)

for class_name in required_classes:
    if not hasattr(bar_ai, class_name):
        fail(f"Fehlende Klasse: {class_name}")

    print(f"✅ Klasse vorhanden: {class_name}")


# ----------------------------------------------------
# Exportierte Shared-Memory-Methoden prüfen
# ----------------------------------------------------

required_shared_memory_methods = (
    "create",
    "open",
    "remove",
    "write_unit_data",
    "write_action",
    "read_all_units",
    "get_own_team_id",
)

for method_name in required_shared_memory_methods:
    if not hasattr(bar_ai.SharedMemory, method_name):
        fail(
            "Fehlende SharedMemory-Methode: "
            f"{method_name}"
        )

    print(
        "✅ SharedMemory-Methode vorhanden: "
        f"{method_name}"
    )


# ----------------------------------------------------
# ActionId testen
# ----------------------------------------------------

try:
    action_id_names = (
        "MoveRight",
        "MoveLeft",
        "MoveUp",
        "MoveDown",
        "Attack",
    )

    print("\n🎮 Prüfe ActionId:")

    for action_name in action_id_names:
        action_id = getattr(
            bar_ai.ActionId,
            action_name,
        )

        print(
            f"✅ ActionId.{action_name} vorhanden: "
            f"{action_id}"
        )

except Exception as e:
    fail("ActionId-Test fehlgeschlagen", e)


# ----------------------------------------------------
# Test-Units erzeugen
#
# Reihenfolge im Shared Memory:
#
# 1. Friendly Agent, Team 1
# 2. Friendly Unit, Team 1
# 3. Sichtbarer Enemy, Team 2
# 4. Sichtbarer Enemy, Team 3
#
# Die erste Unit bestimmt die eigene Team-ID.
# ----------------------------------------------------

try:
    friendly_agent = create_unit(
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

    friendly_unit = create_unit(
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

    visible_enemy_one = create_unit(
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

    visible_enemy_two = create_unit(
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

    print("\n✅ Vier Test-Units erfolgreich erstellt")

except Exception as e:
    fail("Fehler beim Erstellen der Test-Units", e)


# ----------------------------------------------------
# Erste UnitData direkt über das Binding validieren
# ----------------------------------------------------

friendly_agent_expected_values = {
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

try:
    print("\n📊 Validiere Friendly Agent:")

    validate_fields(
        friendly_agent,
        friendly_agent_expected_values,
        "friendly_agent",
    )

except (AttributeError, AssertionError) as e:
    fail("Friendly-Agent-Validierung fehlgeschlagen", e)


# ----------------------------------------------------
# Action erstellen und validieren
# ----------------------------------------------------

try:
    action = bar_ai.Action()

    action.unit_id = 42
    action.team_id = 1
    action.ally_team_id = 0
    action.action_id = bar_ai.ActionId.Attack
    action.target_unit_id = 99

    print("\n✅ Action erstellt")

    action_expected_values = {
        "unit_id": 42,
        "team_id": 1,
        "ally_team_id": 0,
        "action_id": bar_ai.ActionId.Attack,
        "target_unit_id": 99,
    }

    validate_fields(
        action,
        action_expected_values,
        "action",
    )

except Exception as e:
    fail("Action-Test fehlgeschlagen", e)


# ----------------------------------------------------
# Shared Memory vorbereiten
# ----------------------------------------------------

shared_memory_name = (
    f"/bar_ai_test_{os.getpid()}_{time.time_ns()}"
)

shared_memory = None
opened_shared_memory = None
shared_memory_created = False
test_exit_code = 0

print("\n🧠 Starte Shared-Memory-Test")
print("Shared-Memory-Name:", shared_memory_name)


try:
    # ------------------------------------------------
    # Eventuelle Altlast entfernen
    # ------------------------------------------------

    try:
        bar_ai.SharedMemory.remove(
            shared_memory_name
        )
    except Exception:
        # Normal, wenn der Name noch nicht existiert.
        pass

    # ------------------------------------------------
    # Shared Memory erstellen
    # ------------------------------------------------

    shared_memory = bar_ai.SharedMemory.create(
        shared_memory_name
    )

    shared_memory_created = True

    print("✅ Shared Memory erfolgreich erstellt")

    # ------------------------------------------------
    # Friendly Units zuerst schreiben
    # ------------------------------------------------

    friendly_agent_serializable_id = (
        shared_memory.write_unit_data(
            friendly_agent
        )
    )

    friendly_unit_serializable_id = (
        shared_memory.write_unit_data(
            friendly_unit
        )
    )

    print(
        "✅ Friendly Agent geschrieben, ID:",
        friendly_agent_serializable_id,
    )

    print(
        "✅ Friendly Unit geschrieben, ID:",
        friendly_unit_serializable_id,
    )

    # ------------------------------------------------
    # Sichtbare Enemy Units danach schreiben
    # ------------------------------------------------

    enemy_one_serializable_id = (
        shared_memory.write_unit_data(
            visible_enemy_one
        )
    )

    enemy_two_serializable_id = (
        shared_memory.write_unit_data(
            visible_enemy_two
        )
    )

    print(
        "✅ Enemy 1 geschrieben, ID:",
        enemy_one_serializable_id,
    )

    print(
        "✅ Enemy 2 geschrieben, ID:",
        enemy_two_serializable_id,
    )

    # ------------------------------------------------
    # Action schreiben
    # ------------------------------------------------

    action_serializable_id = (
        shared_memory.write_action(action)
    )

    print(
        "✅ Action geschrieben, ID:",
        action_serializable_id,
    )

    # ------------------------------------------------
    # Serializable-IDs validieren
    # ------------------------------------------------

    serializable_ids = [
        friendly_agent_serializable_id,
        friendly_unit_serializable_id,
        enemy_one_serializable_id,
        enemy_two_serializable_id,
        action_serializable_id,
    ]

    assert all(
        isinstance(serializable_id, int)
        for serializable_id in serializable_ids
    ), (
        "Alle write-Funktionen müssen Integer-IDs "
        f"zurückgeben: {serializable_ids!r}"
    )

    assert len(set(serializable_ids)) == 5, (
        "Serializable-IDs sind nicht eindeutig: "
        f"{serializable_ids!r}"
    )

    assert serializable_ids == [0, 1, 2, 3, 4], (
        "Unerwartete Serializable-ID-Reihenfolge. "
        f"Erwartet [0, 1, 2, 3, 4], "
        f"erhalten {serializable_ids!r}"
    )

    print(
        "✅ Serializable-IDs sind eindeutig und "
        "korrekt sortiert"
    )

    # ------------------------------------------------
    # Eigene Team-ID prüfen
    # ------------------------------------------------

    own_team_id = (
        shared_memory.get_own_team_id()
    )

    assert own_team_id == 1, (
        "Die erste Unit sollte Team-ID 1 festlegen, "
        f"erhalten: {own_team_id}"
    )

    print(
        "✅ Eigene Team-ID:",
        own_team_id,
    )

    # ------------------------------------------------
    # Alle UnitData-Objekte lesen
    # ------------------------------------------------

    units = shared_memory.read_all_units()

    assert isinstance(units, list), (
        "read_all_units() sollte eine Python-Liste "
        f"zurückgeben, erhalten: {type(units).__name__}"
    )

    assert len(units) == 4, (
        "read_all_units() sollte genau vier UnitData-Objekte "
        "zurückgeben. Der Action-Eintrag muss übersprungen "
        f"werden. Erhalten: {len(units)}"
    )

    read_unit_ids = unit_ids(units)

    assert read_unit_ids == [42, 43, 99, 100], (
        "Falsche Unit-Reihenfolge. "
        f"Erwartet [42, 43, 99, 100], "
        f"erhalten {read_unit_ids!r}"
    )

    print(
        "✅ read_all_units() liefert vier Units:",
        read_unit_ids,
    )

    # ------------------------------------------------
    # Team-Reihenfolge prüfen
    # ------------------------------------------------

    read_team_ids = [
        unit.team_id
        for unit in units
    ]

    assert read_team_ids == [1, 1, 2, 3], (
        "Falsche Team-Reihenfolge. "
        f"Erwartet [1, 1, 2, 3], "
        f"erhalten {read_team_ids!r}"
    )

    print(
        "✅ Team-Reihenfolge korrekt:",
        read_team_ids,
    )

    # ------------------------------------------------
    # Typen der gelesenen Objekte prüfen
    # ------------------------------------------------

    for index, unit in enumerate(units):
        assert isinstance(unit, bar_ai.UnitData), (
            f"Element {index} ist kein UnitData-Objekt: "
            f"{type(unit).__name__}"
        )

    print(
        "✅ Alle gelesenen Elemente sind UnitData-Objekte"
    )

    # ------------------------------------------------
    # Inhalt der gelesenen Units prüfen
    # ------------------------------------------------

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

    for index, expected_values in enumerate(
        expected_units
    ):
        validate_fields(
            units[index],
            expected_values,
            f"units[{index}]",
        )

    print(
        "✅ Inhalte aller gelesenen Units sind korrekt"
    )

    # ------------------------------------------------
    # Friendly/Enemy-Aufteilung in Python prüfen
    # ------------------------------------------------

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

    assert unit_ids(friendly_units) == [42, 43], (
        "Falsche Friendly Units: "
        f"{unit_ids(friendly_units)!r}"
    )

    assert unit_ids(enemy_units) == [99, 100], (
        "Falsche Enemy Units: "
        f"{unit_ids(enemy_units)!r}"
    )

    print(
        "✅ Friendly Units:",
        unit_ids(friendly_units),
    )

    print(
        "✅ Sichtbare Enemy Units:",
        unit_ids(enemy_units),
    )

    # ------------------------------------------------
    # Shared Memory durch ein zweites Objekt öffnen
    # ------------------------------------------------

    opened_shared_memory = (
        bar_ai.SharedMemory.open(
            shared_memory_name
        )
    )

    print(
        "✅ Vorhandenes Shared Memory erfolgreich geöffnet"
    )

    # ------------------------------------------------
    # Eigene Team-ID nach open() prüfen
    # ------------------------------------------------

    opened_own_team_id = (
        opened_shared_memory.get_own_team_id()
    )

    assert opened_own_team_id == 1, (
        "Nach open() wurde eine falsche Team-ID "
        f"ermittelt: {opened_own_team_id}"
    )

    print(
        "✅ Eigene Team-ID nach open():",
        opened_own_team_id,
    )

    # ------------------------------------------------
    # Alle Units nach open() erneut lesen
    # ------------------------------------------------

    reopened_units = (
        opened_shared_memory.read_all_units()
    )

    reopened_unit_ids = unit_ids(
        reopened_units
    )

    assert reopened_unit_ids == [42, 43, 99, 100], (
        "Nach open() wurden falsche Units gelesen. "
        f"Erwartet [42, 43, 99, 100], "
        f"erhalten {reopened_unit_ids!r}"
    )

    assert len(reopened_units) == 4, (
        "Nach open() sollte read_all_units() vier "
        f"Units liefern, erhalten: {len(reopened_units)}"
    )

    print(
        "✅ Units nach open():",
        reopened_unit_ids,
    )

    # Nicht über opened_shared_memory schreiben.
    # Der IdAllocator wird nach open() aktuell noch nicht
    # zuverlässig aus den vorhandenen IDs rekonstruiert.

except Exception as e:
    print("❌ Shared-Memory-Test fehlgeschlagen:")
    print(f"{type(e).__name__}: {e}")
    test_exit_code = 1

finally:
    # C++-Objekte und mmap-Verbindungen freigeben.
    opened_shared_memory = None
    shared_memory = None

    if shared_memory_created:
        try:
            bar_ai.SharedMemory.remove(
                shared_memory_name
            )

            print(
                "✅ Shared Memory erfolgreich entfernt"
            )

        except Exception as cleanup_error:
            print(
                "⚠️ Shared Memory konnte nicht "
                "entfernt werden:"
            )

            print(
                f"{type(cleanup_error).__name__}: "
                f"{cleanup_error}"
            )

            test_exit_code = 1


if test_exit_code != 0:
    sys.exit(test_exit_code)


print(
    "\n🎉 bar_ai UnitData-, Action- und "
    "Shared-Memory-Test erfolgreich abgeschlossen!"
)