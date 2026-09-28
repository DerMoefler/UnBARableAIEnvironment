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
    unit_def_name,
    human_name,
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
    unit.unit_def_name = unit_def_name
    unit.human_name = human_name

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
    """Vergleicht die Python-Attribute eines gebundenen C++-Objekts."""
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
    """Extrahiert Unit-IDs aus einer Liste von UnitData-Objekten."""
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
# Prüfen, ob Klassen exportiert wurden
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
# Prüfen, ob Shared-Memory-Methoden exportiert wurden
# ----------------------------------------------------

required_shared_memory_methods = (
    "create",
    "open",
    "remove",
    "write_unit_data",
    "write_action",
    "get_unit_by_id",
    "get_enemy_units_in_sight",
    "get_ally_units_in_sight",
    "get_ally_unit_IDs",
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
        action_id = getattr(bar_ai.ActionId, action_name)

        print(
            f"✅ ActionId.{action_name} vorhanden: "
            f"{action_id}"
        )

except Exception as e:
    fail("ActionId-Test fehlgeschlagen", e)


# ----------------------------------------------------
# Test-Units erzeugen
#
# Speicherreihenfolge:
#
# 1. Friendly Agent, team_id 1
# 2. Friendly Unit, team_id 1
# 3. Sichtbarer Enemy, team_id 2
# 4. Sichtbarer Enemy, team_id 3
#
# Die erste Unit bestimmt die eigene team_id.
# ----------------------------------------------------

try:
    friendly_agent = create_unit(
        unit_id=42,
        team_id=1,
        ally_team_id=0,
        unit_def_id=101,
        unit_def_name="armcom",
        human_name="Armada Commander",
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
        unit_def_name="armmex",
        human_name="Metal Extractor",
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
        unit_def_name="corcom",
        human_name="Cortex Commander",
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
        unit_def_name="corak",
        human_name="Grunt",
        pos_x=900.0,
        pos_y=20.0,
        pos_z=750.0,
        health=75.0,
        max_health=100.0,
    )

    print("\n✅ Test-Units erfolgreich erstellt")

except Exception as e:
    fail("Fehler beim Erstellen der Test-Units", e)


# ----------------------------------------------------
# Friendly Agent validieren
# ----------------------------------------------------

friendly_agent_expected_values = {
    "unit_id": 42,
    "unit_def_id": 101,
    "unit_def_name": "armcom",
    "human_name": "Armada Commander",
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

    print("\n✅ Action erstellt und beschrieben")

    print("\n📊 Action:")
    print("unit_id:", action.unit_id)
    print("team_id:", action.team_id)
    print("ally_team_id:", action.ally_team_id)
    print("action_id:", action.action_id)
    print("target_unit_id:", action.target_unit_id)

except Exception as e:
    fail("Fehler beim Erstellen der Action", e)


action_expected_values = {
    "unit_id": 42,
    "team_id": 1,
    "ally_team_id": 0,
    "action_id": bar_ai.ActionId.Attack,
    "target_unit_id": 99,
}

try:
    print("\n📊 Validiere Action:")

    validate_fields(
        action,
        action_expected_values,
        "action",
    )

except (AttributeError, AssertionError) as e:
    fail("Action-Validierung fehlgeschlagen", e)


# ----------------------------------------------------
# Shared Memory vorbereiten
# ----------------------------------------------------

# Prozess-ID und Zeitstempel verhindern Namenskonflikte.
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
        bar_ai.SharedMemory.remove(shared_memory_name)
        print("ℹ️ Vorhandenes Shared Memory wurde entfernt")
    except Exception:
        # Der Shared-Memory-Name existiert normalerweise noch nicht.
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
        shared_memory.write_unit_data(friendly_agent)
    )

    friendly_unit_serializable_id = (
        shared_memory.write_unit_data(friendly_unit)
    )

    print(
        "✅ Friendly Agent geschrieben, Serializable-ID:",
        friendly_agent_serializable_id,
    )

    print(
        "✅ Friendly Unit geschrieben, Serializable-ID:",
        friendly_unit_serializable_id,
    )

    # ------------------------------------------------
    # Danach bereits sichtbare Enemy Units schreiben
    # ------------------------------------------------

    enemy_one_serializable_id = (
        shared_memory.write_unit_data(visible_enemy_one)
    )

    enemy_two_serializable_id = (
        shared_memory.write_unit_data(visible_enemy_two)
    )

    print(
        "✅ Sichtbarer Enemy 1 geschrieben, Serializable-ID:",
        enemy_one_serializable_id,
    )

    print(
        "✅ Sichtbarer Enemy 2 geschrieben, Serializable-ID:",
        enemy_two_serializable_id,
    )

    # ------------------------------------------------
    # Action nach den UnitData-Einträgen schreiben
    # ------------------------------------------------

    action_serializable_id = shared_memory.write_action(
        action
    )

    print(
        "✅ Action geschrieben, Serializable-ID:",
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
        "Alle write-Funktionen müssen Integer-IDs zurückgeben. "
        f"Erhalten: {serializable_ids!r}"
    )

    assert all(
        serializable_id >= 0
        for serializable_id in serializable_ids
    ), (
        "Alle Serializable-IDs müssen nichtnegativ sein. "
        f"Erhalten: {serializable_ids!r}"
    )

    assert len(set(serializable_ids)) == len(serializable_ids), (
        "Serializable-IDs sind nicht eindeutig: "
        f"{serializable_ids!r}"
    )

    assert serializable_ids == [0, 1, 2, 3, 4], (
        "Unerwartete Reihenfolge der Serializable-IDs. "
        f"Erwartet [0, 1, 2, 3, 4], "
        f"erhalten {serializable_ids!r}"
    )

    print("✅ Serializable-IDs sind eindeutig und korrekt sortiert")

    # ------------------------------------------------
    # Eigene Team-ID testen
    # ------------------------------------------------

    own_team_id = shared_memory.get_own_team_id()

    assert own_team_id == 1, (
        "Die erste Unit sollte team_id 1 als eigenes Team "
        f"festlegen, erhalten: {own_team_id}"
    )

    print(
        "✅ Eigene Team-ID wurde aus der ersten Unit bestimmt:",
        own_team_id,
    )

    # ------------------------------------------------
    # Unit anhand ihrer BAR-Unit-ID lesen
    # ------------------------------------------------

    found_agent = shared_memory.get_unit_by_id(42)

    assert found_agent.unit_id == 42, (
        "get_unit_by_id(42) gab die falsche Unit zurück: "
        f"{found_agent.unit_id}"
    )

    assert found_agent.team_id == 1, (
        "Der gelesene Agent besitzt die falsche team_id: "
        f"{found_agent.team_id}"
    )

    assert found_agent.unit_def_name == "armcom", (
        "Der gelesene Agent besitzt den falschen Namen: "
        f"{found_agent.unit_def_name!r}"
    )

    print("✅ get_unit_by_id(42) liefert den Friendly Agent")

    found_enemy = shared_memory.get_unit_by_id(99)

    assert found_enemy.unit_id == 99
    assert found_enemy.team_id == 2
    assert found_enemy.unit_def_name == "corcom"

    print("✅ get_unit_by_id(99) liefert Enemy 1")

    # ------------------------------------------------
    # Friendly Unit-IDs testen
    # ------------------------------------------------

    ally_ids = shared_memory.get_ally_unit_IDs()

    assert ally_ids == [42, 43], (
        "Unerwartete Friendly Unit-IDs. "
        f"Erwartet [42, 43], erhalten {ally_ids!r}"
    )

    print("✅ Friendly Unit-IDs:", ally_ids)

    # ------------------------------------------------
    # Friendly Units für Agent 42 testen
    # ------------------------------------------------

    ally_units = (
        shared_memory.get_ally_units_in_sight(42)
    )

    ally_ids_in_sight = unit_ids(ally_units)

    # Der Agent selbst wird von der C++-Methode ausgeschlossen.
    assert ally_ids_in_sight == [43], (
        "Unerwartete Friendly Units für Agent 42. "
        f"Erwartet [43], erhalten {ally_ids_in_sight!r}"
    )

    print(
        "✅ Friendly Units für Agent 42:",
        ally_ids_in_sight,
    )

    # ------------------------------------------------
    # Bereits sichtbare Enemy Units testen
    # ------------------------------------------------

    enemy_units = (
        shared_memory.get_enemy_units_in_sight(42)
    )

    enemy_ids_in_sight = unit_ids(enemy_units)

    # Es findet keine Distanz- oder LOS-Berechnung statt.
    # Alle nach den Friendly Units gespeicherten Enemy Units
    # gelten bereits als sichtbar.
    assert enemy_ids_in_sight == [99, 100], (
        "Unerwartete sichtbare Enemy Units. "
        f"Erwartet [99, 100], "
        f"erhalten {enemy_ids_in_sight!r}"
    )

    print(
        "✅ Bereits sichtbare Enemy Units:",
        enemy_ids_in_sight,
    )

    # ------------------------------------------------
    # Fehlerfall: unbekannte Unit-ID
    # ------------------------------------------------

    try:
        shared_memory.get_unit_by_id(999999)
    except (IndexError, KeyError, RuntimeError) as expected_error:
        print(
            "✅ Unbekannte Unit-ID löst erwarteten Fehler aus:",
            type(expected_error).__name__,
        )
    else:
        raise AssertionError(
            "get_unit_by_id(999999) hätte einen Fehler "
            "auslösen müssen."
        )

    # ------------------------------------------------
    # Fehlerfall: Enemy darf nicht als eigener Agent gelten
    # ------------------------------------------------

    try:
        shared_memory.get_enemy_units_in_sight(99)
    except (ValueError, RuntimeError) as expected_error:
        print(
            "✅ Enemy-ID als Agent wird korrekt abgelehnt:",
            type(expected_error).__name__,
        )
    else:
        raise AssertionError(
            "get_enemy_units_in_sight(99) hätte einen Fehler "
            "auslösen müssen, weil Unit 99 nicht zum eigenen "
            "Team gehört."
        )

    # ------------------------------------------------
    # Vorhandenes Shared Memory erneut öffnen
    # ------------------------------------------------

    opened_shared_memory = bar_ai.SharedMemory.open(
        shared_memory_name
    )

    print("✅ Vorhandenes Shared Memory erfolgreich geöffnet")
    print("✅ Layout-Tabelle wurde initialisiert")

    # ------------------------------------------------
    # Team-ID nach open() rekonstruieren
    # ------------------------------------------------

    opened_own_team_id = (
        opened_shared_memory.get_own_team_id()
    )

    assert opened_own_team_id == 1, (
        "Nach open() wurde eine falsche eigene Team-ID "
        f"ermittelt: {opened_own_team_id}"
    )

    print(
        "✅ Eigene Team-ID nach open():",
        opened_own_team_id,
    )

    # ------------------------------------------------
    # Abfragen nach open() erneut testen
    # ------------------------------------------------

    opened_ally_ids = (
        opened_shared_memory.get_ally_unit_IDs()
    )

    opened_ally_units = (
        opened_shared_memory.get_ally_units_in_sight(42)
    )

    opened_enemy_units = (
        opened_shared_memory.get_enemy_units_in_sight(42)
    )

    assert opened_ally_ids == [42, 43], (
        "Falsche Friendly IDs nach open(): "
        f"{opened_ally_ids!r}"
    )

    assert unit_ids(opened_ally_units) == [43], (
        "Falsche Friendly Units nach open(): "
        f"{unit_ids(opened_ally_units)!r}"
    )

    assert unit_ids(opened_enemy_units) == [99, 100], (
        "Falsche Enemy Units nach open(): "
        f"{unit_ids(opened_enemy_units)!r}"
    )

    print("✅ Friendly IDs nach open():", opened_ally_ids)
    print(
        "✅ Friendly Units nach open():",
        unit_ids(opened_ally_units),
    )
    print(
        "✅ Enemy Units nach open():",
        unit_ids(opened_enemy_units),
    )

    # Aktuell absichtlich nicht über opened_shared_memory schreiben.
    #
    # initializeLayouts() muss den IdAllocator korrekt wiederherstellen,
    # bevor nach open() neue Objekte sicher geschrieben werden können.

except Exception as e:
    print("❌ Shared-Memory-Test fehlgeschlagen:")
    print(f"{type(e).__name__}: {e}")
    test_exit_code = 1

finally:
    # Zuerst die Python-Referenzen freigeben. Dadurch können die
    # C++-Destruktoren ihre Shared-Memory-Mappings schließen.
    opened_shared_memory = None
    shared_memory = None

    if shared_memory_created:
        try:
            bar_ai.SharedMemory.remove(shared_memory_name)
            print("✅ Shared Memory erfolgreich entfernt")
        except Exception as cleanup_error:
            print("⚠️ Shared Memory konnte nicht entfernt werden:")
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