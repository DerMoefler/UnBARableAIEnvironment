#include "memory/bar_shared_memory.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace UnBARableAINS::memory {

BarSharedMemory::BarSharedMemory(Impl impl)
    : m_sharedMemory(std::move(impl)) {}

BarSharedMemory BarSharedMemory::create(std::string_view name) {
    return BarSharedMemory{
        Impl::create(name)
    };
}

BarSharedMemory BarSharedMemory::open(std::string_view name) {
    return BarSharedMemory{
        Impl::open(name)
    };
}

void BarSharedMemory::remove(std::string_view name) {
    Impl::remove(std::string{name});
}

BarSharedMemory::SerializableId BarSharedMemory::writeAction(const Action& action) {
    return m_sharedMemory.write(action);
}

BarSharedMemory::SerializableId BarSharedMemory::writeUnitData(const UnitData& unitData) {
    /*
     * Die erste über diesen Wrapper geschriebene UnitData bestimmt
     * die eigene Team-ID.
     */
    if (!m_ownTeamId.has_value()) {
        m_ownTeamId = unitData.team_id;
    }

    /*
     * Optionaler Konsistenzhinweis:
     *
     * Friendly Units sollen zuerst geschrieben werden. Sobald Enemy Units
     * geschrieben werden, sollte danach keine Friendly Unit mehr folgen.
     *
     * Diese Reihenfolge kann hier ohne zusätzlichen Zustand noch nicht
     * vollständig geprüft werden.
     */
    return m_sharedMemory.write(unitData);
}

BarSharedMemory::UnitData BarSharedMemory::getUnitById(int unitId) {
    const std::vector<UnitData> units = readAllUnits();

    for (const UnitData& unit : units) {
        if (unit.unit_id == unitId) {
            return unit;
        }
    }

    throw std::out_of_range(
        "No UnitData found for unit ID " +
        std::to_string(unitId));
}

std::vector<BarSharedMemory::UnitData> BarSharedMemory::getEnemyUnitsInSight(int agentId) {
    /*
     * Prüft, dass der Agent existiert und eine eigene Unit ist.
     *
     * Die Sichtweite wird nicht berechnet. Alle gegnerischen Units,
     * die im Shared Memory stehen, gelten bereits als sichtbar.
     */
    const UnitData agent = getOwnAgentById(agentId);
    const int ownTeamId = agent.team_id;

    const std::vector<UnitData> units = readAllUnits();

    std::vector<UnitData> enemies;
    bool enemySectionReached = false;

    for (const UnitData& unit : units) {
        if (unit.team_id != ownTeamId) {
            enemies.push_back(unit)
        }
    }

    return enemies;
}

std::vector<BarSharedMemory::UnitData> BarSharedMemory::getAllyUnitsInSight(int agentId) {
    const UnitData agent = getOwnAgentById(agentId);
    const int ownTeamId = agent.team_id;

    const std::vector<UnitData> units = readAllUnits();

    std::vector<UnitData> allies;

    for (const UnitData& unit : units) {
        /*
         * Die erste Unit mit einer anderen team_id markiert den Beginn
         * des Enemy-Abschnitts.
         */
        if (unit.team_id != ownTeamId) {
            break;
        }

        /*
         * Der Agent selbst soll nicht in seiner Ally-Liste stehen.
         */
        if (unit.unit_id == agentId) {
            continue;
        }

        allies.push_back(unit);
    }

    return allies;
}

std::vector<int> BarSharedMemory::getAllyUnitIds() {
    const int ownTeamId = resolveOwnTeamId();
    const std::vector<UnitData> units = readAllUnits();

    std::vector<int> allyUnitIds;
    allyUnitIds.reserve(units.size());

    for (const UnitData& unit : units) {
        /*
         * Alle Friendly Units stehen am Anfang.
         * Beim ersten Enemy-Eintrag kann die Suche beendet werden.
         */
        if (unit.team_id != ownTeamId) {
            break;
        }

        allyUnitIds.push_back(unit.unit_id);
    }

    return allyUnitIds;
}

int BarSharedMemory::getOwnTeamId() {
    return resolveOwnTeamId();
}

int BarSharedMemory::resolveOwnTeamId() {
    if (m_ownTeamId.has_value()) {
        return m_ownTeamId.value();
    }

    /*
     * Nach BarSharedMemory::open() wurde über dieses C++-Objekt noch
     * keine UnitData geschrieben. Daher wird die erste gespeicherte
     * UnitData gelesen.
     */
    const std::vector<UnitData> units = readAllUnits();

    if (units.empty()) {
        throw std::runtime_error(
            "Cannot determine own team ID because no UnitData "
            "exists in shared memory.");
    }

    m_ownTeamId = units.front().team_id;

    return m_ownTeamId.value();
}

BarSharedMemory::UnitData BarSharedMemory::getOwnAgentById(int agentId) {
    const UnitData agent = getUnitById(agentId);
    const int ownTeamId = resolveOwnTeamId();

    if (agent.team_id != ownTeamId) {
        throw std::invalid_argument(
            "Unit with ID " +
            std::to_string(agentId) +
            " does not belong to the own team.");
    }

    if (agent.is_dead) {
        throw std::invalid_argument(
            "Unit with ID " +
            std::to_string(agentId) +
            " is dead.");
    }

    return agent;
}

std::vector<BarSharedMemory::UnitData> BarSharedMemory::readAllUnits() {
    std::vector<UnitData> units;

    SerializableId serializableId = 0;

    while (true) {
        try {
            auto& layoutVariant =
                m_sharedMemory.getLayout(serializableId);

            /*
             * Das Shared Memory kann sowohl UnitData als auch Action
             * enthalten. Nur UnitData-Einträge werden gelesen.
             */
            const bool containsUnitData =
                std::holds_alternative<
                    serialization::Layout<UnitData>
                >(layoutVariant);

            if (containsUnitData) {
                /*
                 * TODO:
                 * SharedMemory::read<S>(serializableId) existiert
                 * aktuell noch nicht.
                 *
                 * Erwartete Signatur in shared_memory.h:
                 *
                 * template <serialization::Serializable S>
                 * S read(id::id_t serializableId);
                 */
                UnitData unit =
                    m_sharedMemory.template read<UnitData>(
                        serializableId);

                units.push_back(std::move(unit));
            }
        }
        catch (const std::out_of_range&) {
            /*
             * getLayout() wirft std::out_of_range, wenn serializableId
             * nicht mehr in m_layouts vorhanden ist.
             */
            break;
        }

        ++serializableId;
    }

    return units;
}

}  // namespace UnBARableAINS::memory