#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "action.h"
#include "unit_data.h"

#include "../../src/memory/shared_memory.h"
#include "../../src/memory/shared_memory_posix.h"

namespace UnBARableAINS::memory {

/**
 * BAR-spezifische Fassade für das generische Shared-Memory-System.
 *
 * Erwartete Reihenfolge der UnitData-Objekte:
 *
 * 1. Alle Units des eigenen Teams
 * 2. Alle bereits sichtbaren gegnerischen Units
 *
 * Die erste gespeicherte UnitData bestimmt die eigene team_id.
 */
class BarSharedMemory {
public:
    using UnitData = UnBARableAINS::unit::UnitData;
    using Action = UnBARableAINS::Action;

    using Impl = SharedMemory<
        SharedMemoryPosix,
        UnitData,
        Action>;

    using SerializableId = UnBARableAINS::id::id_t;

    /**
     * Erstellt einen neuen POSIX-Shared-Memory-Bereich.
     */
    static BarSharedMemory create(std::string_view name);

    /**
     * Öffnet einen vorhandenen POSIX-Shared-Memory-Bereich.
     */
    static BarSharedMemory open(std::string_view name);

    /**
     * Entfernt einen POSIX-Shared-Memory-Namen.
     */
    static void remove(std::string_view name);

    BarSharedMemory(const BarSharedMemory&) = delete;
    BarSharedMemory& operator=(const BarSharedMemory&) = delete;

    BarSharedMemory(BarSharedMemory&&) noexcept = default;
    BarSharedMemory& operator=(BarSharedMemory&&) noexcept = default;

    ~BarSharedMemory() = default;

    /**
     * Schreibt eine Action in das Shared Memory.
     */
    SerializableId writeAction(const Action& action);

    /**
     * Schreibt UnitData in das Shared Memory.
     *
     * Die team_id der ersten UnitData wird als eigene Team-ID gespeichert.
     */
    SerializableId writeUnitData(const UnitData& unitData);

    /**
     * Liest alle UnitData-Objekte in Serializable-ID-Reihenfolge.
     *
     * TODO:
     * Benötigt SharedMemory::read<UnitData>(serializableId).
     */
    std::vector<UnitData> readAllUnits();

    /**
     * Liefert die Team-ID der ersten gespeicherten UnitData.
     *
     * @throws std::runtime_error, wenn keine UnitData vorhanden ist.
     */
    int getOwnTeamId();

private:
    explicit BarSharedMemory(Impl impl);

    /**
     * Liefert die eigene Team-ID.
     *
     * Wenn die Team-ID noch nicht bekannt ist, wird die erste
     * UnitData aus dem Shared Memory gelesen.
     */
    int resolveOwnTeamId();

    /**
     * Prüft, ob agentId eine lebende Unit des eigenen Teams bezeichnet.
     *
     * @throws std::out_of_range, wenn die Unit nicht existiert.
     * @throws std::invalid_argument, wenn die Unit nicht zum eigenen Team gehört.
     */
    UnitData getOwnAgentById(int agentId);

    Impl m_sharedMemory;

    /**
     * Wird beim ersten writeUnitData() gesetzt.
     *
     * Nach open() wird die Team-ID bei Bedarf aus der ersten
     * gespeicherten UnitData rekonstruiert.
     */
    std::optional<int> m_ownTeamId;
};

}  // namespace UnBARableAINS::memory