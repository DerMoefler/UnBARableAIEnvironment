import sys

print("🚀 Starte bar_ai UnitData- und Action-Test...")

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

for class_name in ("UnitData", "Action"):
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
    bar_data.capture_progress = 10
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
    "capture_progress": 10,
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
# Action testen
# ----------------------------------------------------

try:
    action = bar_ai.Action()

    action.unit_id = 42
    action.team_id = 1
    action.ally_team_id = 0
    action.action_id = 5
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
    "action_id": 5,
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

print("\n🎉 bar_ai UnitData- und Action-Test erfolgreich abgeschlossen!")