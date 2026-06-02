print("🚀 Starte unbarableaienvironment Test...")

try:
    import unbarableaienvironment
    print("✅ import unbarableaienvironment erfolgreich")
except Exception as e:
    print("❌ import unbarableaienvironment fehlgeschlagen:")
    print(e)
    exit(1)

print("📦 Modul:", unbarableaienvironment)

# ----------------------------------------------------
# Teste ob Klassen existieren
# ----------------------------------------------------

required = ["UnitData", "Pawn"]

for name in required:
    if not hasattr(unbarableaienvironment, name):
        print(f"❌ Fehlende Klasse: {name}")
        exit(1)
    else:
        print(f"✅ Klasse vorhanden: {name}")

# ----------------------------------------------------
# Teste Instanziierung + Methoden
# ----------------------------------------------------

try:
    data = unbarableaienvironment.UnitData()
    data.health = 100.0
    data.team = 1
    data.xPosition = 10.0
    data.yPosition = 20.0
    data.zPosition = 5.0
    data.hasCurrentCommand = True

    print("✅ UnitData erstellt")

    pawn = unbarableaienvironment.Pawn(data)
    print("✅ Pawn erstellt")

    print("📊 Werte:")
    print("health:", pawn.getHealth())
    print("team:", pawn.getTeam())
    print("pos:", pawn.getXPosition(), pawn.getYPosition(), pawn.getZPosition())
    print("has command:", pawn.hasCurrentCommand())

    print("✅ Methoden funktionieren")

except Exception as e:
    print("❌ Fehler bei Pawn / Methoden:")
    print(e)
    exit(1)

# ----------------------------------------------------
# Fertig
# ----------------------------------------------------

print("🎉 unbarableaienvironment Test erfolgreich abgeschlossen!")
