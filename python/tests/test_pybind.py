import os
import sys
import time

print("🚀 Starte bar_ai UnitData-, Action- und Shared-Memory-Test...")

# ----------------------------------------------------
# Modul importieren
# ----------------------------------------------------

try:
    import bar_ai
    print("✅ import bar_ai erfolgreich")
except Exception as e:
    print("❌ import bar_ai fehlgeschlagen:")
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)

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
        print(f"❌ Fehlende Klasse: {class_name}")
        sys.exit(1)

    print(f"✅ Klasse vorhanden: {class_name}")

# ----------------------------------------------------
# UnitData testen
# ----------------------------------------------------

try:
    bar_data = bar_ai.UnitData()

    # Identifikation
    bar_data.unit_id = 42
    bar_data.unit_def_id = 3
    bar_data.unit_def_name = "armmex"
    bar_data.human_name = "Metal Extractor"

    # Team
    bar_data.team_id = 1
    bar_data.ally_team_id = 0

    # Lebenspunkte
    bar_data.health = 100.0
    bar_data.max_health = 250.0

    # Position
    bar_data.pos_x = 10.0
    bar_data.pos_y = 20.0
    bar_data.pos_z = 5.0

    # Sichtweite
    bar_data.los_radius = 500.0
    bar_data.air_los_radius = 350.0

    # Status
    bar_data.is_dead = False
    bar_data.being_built = True

    # Fortschritt und Schaden
    bar_data.build_progress = 0.75
    bar_data.capture_progress = 10.0
    bar_data.paralyze_damage = 15.0

    print("✅ UnitData erstellt und beschrieben")

    print("\n📊 UnitData:")
    print("unit_id:", bar_data.unit_id)
    print("unit_def_id:", bar_data.unit_def_id)
    print("unit_def_name:", bar_data.unit_def_name)
    print("human_name:", bar_data.human_name)

    print("team_id:", bar_data.team_id)
    print("ally_team_id:", bar_data.ally_team_id)

    print("health:", bar_data.health)
    print("max_health:", bar_data.max_health)

    print(
        "position:",
        bar_data.pos_x,
        bar_data.pos_y,
        bar_data.pos_z,
    )

    print("los_radius:", bar_data.los_radius)
    print("air_los_radius:", bar_data.air_los_radius)

    print("is_dead:", bar_data.is_dead)
    print("being_built:", bar_data.being_built)

    print("build_progress:", bar_data.build_progress)
    print("capture_progress:", bar_data.capture_progress)
    print("paralyze_damage:", bar_data.paralyze_damage)

except Exception as e:
    print("❌ Fehler beim Erstellen oder Auslesen von UnitData:")
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)

# ----------------------------------------------------
# UnitData validieren
# ----------------------------------------------------

unit_expected_values = {
    "unit_id": 42,
    "unit_def_id": 3,
    "unit_def_name": "armmex",
    "human_name": "Metal Extractor",
    "team_id": 1,
    "ally_team_id": 0,
    "health": 100.0,
    "max_health": 250.0,
    "pos_x": 10.0,
    "pos_y": 20.0,
    "pos_z": 5.0,
    "los_radius": 500.0,
    "air_los_radius": 350.0,
    "is_dead": False,
    "being_built": True,
    "build_progress": 0.75,
    "capture_progress": 10.0,
    "paralyze_damage": 15.0,
}

try:
    for field_name, expected_value in unit_expected_values.items():
        actual_value = getattr(bar_data, field_name)

        assert actual_value == expected_value, (
            f"{field_name}: Erwartet {expected_value!r}, "
            f"erhalten {actual_value!r}"
        )

        print(f"✅ {field_name}: {actual_value!r}")

except (AttributeError, AssertionError) as e:
    print("❌ UnitData-Validierung fehlgeschlagen:")
    print(e)
    sys.exit(1)

# ----------------------------------------------------
# ActionId testen
# ----------------------------------------------------

try:
    action_ids = {
        "MoveRight": 1,
        "MoveLeft": 2,
        "MoveUp": 3,
        "MoveDown": 4,
        "Attack": 5,
    }

    print("\n🎮 Prüfe ActionId:")

    for action_name, expected_value in action_ids.items():
        action_id = getattr(bar_ai.ActionId, action_name)

        print(
            f"✅ ActionId.{action_name} vorhanden: "
            f"{action_id}"
        )

    attack_action_id = bar_ai.ActionId.Attack

except Exception as e:
    print("❌ ActionId-Test fehlgeschlagen:")
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)

# ----------------------------------------------------
# Action testen
# ----------------------------------------------------

try:
    action = bar_ai.Action()

    action.unit_id = 42
    action.team_id = 1
    action.ally_team_id = 0

    # Da Action::action_id vom Typ ActionId ist, wird hier der
    # gebundene Enum-Wert verwendet und nicht direkt die Zahl 5.
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
    print("❌ Fehler beim Erstellen oder Auslesen von Action:")
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)

# ----------------------------------------------------
# Action validieren
# ----------------------------------------------------

action_expected_values = {
    "unit_id": 42,
    "team_id": 1,
    "ally_team_id": 0,
    "action_id": bar_ai.ActionId.Attack,
    "target_unit_id": 99,
}

try:
    for field_name, expected_value in action_expected_values.items():
        actual_value = getattr(action, field_name)

        assert actual_value == expected_value, (
            f"{field_name}: Erwartet {expected_value!r}, "
            f"erhalten {actual_value!r}"
        )

        print(f"✅ {field_name}: {actual_value!r}")

except (AttributeError, AssertionError) as e:
    print("❌ Action-Validierung fehlgeschlagen:")
    print(e)
    sys.exit(1)

# ----------------------------------------------------
# Shared Memory testen
# ----------------------------------------------------

# POSIX-Shared-Memory-Namen sollten mit "/" beginnen.
# Die Prozess-ID verhindert Konflikte zwischen Testläufen.
shared_memory_name = f"/bar_ai_test_{os.getpid()}"

shared_memory = None
opened_shared_memory = None
shared_memory_created = False

print("\n🧠 Starte Shared-Memory-Test")
print("Shared-Memory-Name:", shared_memory_name)

try:
    # Überbleibsel eines vorherigen fehlgeschlagenen Testlaufs entfernen.
    try:
        bar_ai.SharedMemory.remove(shared_memory_name)
        print("ℹ️ Vorhandenes Shared Memory wurde entfernt")
    except Exception:
        # Es ist normal, wenn der Name noch nicht existiert.
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
    # UnitData schreiben
    # ------------------------------------------------

    unit_serializable_id = shared_memory.write_unit_data(
        bar_data
    )

    print(
        "✅ UnitData in Shared Memory geschrieben, "
        f"Serializable-ID: {unit_serializable_id}"
    )

    assert isinstance(unit_serializable_id, int), (
        "write_unit_data() sollte eine Integer-ID zurückgeben, "
        f"erhalten: {type(unit_serializable_id).__name__}"
    )

    assert unit_serializable_id >= 0, (
        "Ungültige UnitData-ID: "
        f"{unit_serializable_id}"
    )

    # ------------------------------------------------
    # Action schreiben
    # ------------------------------------------------

    action_serializable_id = shared_memory.write_action(
        action
    )

    print(
        "✅ Action in Shared Memory geschrieben, "
        f"Serializable-ID: {action_serializable_id}"
    )

    assert isinstance(action_serializable_id, int), (
        "write_action() sollte eine Integer-ID zurückgeben, "
        f"erhalten: {type(action_serializable_id).__name__}"
    )

    assert action_serializable_id >= 0, (
        "Ungültige Action-ID: "
        f"{action_serializable_id}"
    )

    # ------------------------------------------------
    # IDs validieren
    # ------------------------------------------------

    assert unit_serializable_id != action_serializable_id, (
        "UnitData und Action haben dieselbe Serializable-ID: "
        f"{unit_serializable_id}"
    )

    print("✅ Serializable-IDs sind eindeutig")

    # Bei einem frisch erstellten Shared Memory sollten die IDs
    # mit dem aktuellen IdAllocator bei 0 beginnen.
    assert unit_serializable_id == 0, (
        "Für das erste Objekt wurde ID 0 erwartet, "
        f"erhalten: {unit_serializable_id}"
    )

    assert action_serializable_id == 1, (
        "Für das zweite Objekt wurde ID 1 erwartet, "
        f"erhalten: {action_serializable_id}"
    )

    print("✅ Serializable-IDs entsprechen der erwarteten Reihenfolge")

    # ------------------------------------------------
    # Vorhandenes Shared Memory erneut öffnen
    # ------------------------------------------------

    opened_shared_memory = bar_ai.SharedMemory.open(
        shared_memory_name
    )

    print("✅ Vorhandenes Shared Memory erfolgreich geöffnet")
    print("✅ Layout-Tabelle wurde aus Shared Memory initialisiert")

    # Noch nicht über opened_shared_memory schreiben:
    # initializeLayouts() lädt zwar die Layouts, stellt aber den
    # m_idAllocator aktuell nicht wieder her. Dadurch könnte eine
    # bereits verwendete Serializable-ID erneut vergeben werden.

except Exception as e:
    print("❌ Shared-Memory-Test fehlgeschlagen:")
    print(f"{type(e).__name__}: {e}")
    sys.exit_code = 1

else:
    sys.exit_code = 0

finally:
    # Referenzen freigeben, damit die C++-Destruktoren aufgerufen
    # und die mmap-Bereiche geschlossen werden können.
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

if sys.exit_code != 0:
    sys.exit(sys.exit_code)

print(
    "\n🎉 bar_ai UnitData-, Action- und "
    "Shared-Memory-Test erfolgreich abgeschlossen!"
)