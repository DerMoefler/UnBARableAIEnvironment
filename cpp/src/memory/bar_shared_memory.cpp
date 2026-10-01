#include "../../include/UnBARableAI/bar_shared_memory.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace UnBARableAINS::memory {

BarSharedMemory::BarSharedMemory(Impl impl)
    : m_sharedMemory(std::move(impl)) {}

BarSharedMemory BarSharedMemory::create(std::string_view name) {
    return BarSharedMemory{Impl::create(name)};
}

BarSharedMemory BarSharedMemory::open(std::string_view name) {
    return BarSharedMemory{Impl::open(name)};
}

void BarSharedMemory::remove(std::string_view name) { Impl::remove(std::string{name}); }

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

    return m_sharedMemory.write(unitData);
}

int BarSharedMemory::getOwnTeamId() { return resolveOwnTeamId(); }

std::vector<BarSharedMemory::UnitData> BarSharedMemory::readAllUnits() {
    std::vector<UnitData> units;

    SerializableId serializableId = 0;

    while (true) {
        try {
            auto& layoutVariant = m_sharedMemory.getLayout(serializableId);

            /*
             * Das Shared Memory kann sowohl UnitData als auch Action
             * enthalten. Nur UnitData-Einträge werden gelesen.
             */
            const bool containsUnitData =
                std::holds_alternative<serialization::Layout<UnitData> >(layoutVariant);

            if (containsUnitData) {
                ValueVariant unit = m_sharedMemory.read(serializableId);
                assert(std::holds_alternative<UnitData>(unit) &&
                       "Read Serializable must be of type Unit at this point");

                units.push_back(std::get<UnitData>(std::move(unit));
            }
        } catch (const std::out_of_range&) {
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

std::vector<BarSharedMemory::Action> BarSharedMemory::readAllActions() {
    std::vector<Action> actions;

    SerializableId serializableId = 0;

    while (true) {
        try {
            auto& layoutVariant = m_sharedMemory.getLayout(serializableId);

            /*
             * Das Shared Memory kann sowohl UnitData als auch Action
             * enthalten. Nur Action-Einträge werden gelesen.
             */
            const bool containsAction =
                std::holds_alternative<serialization::Layout<Action> >(layoutVariant);

            if (containsAction) {
                ValueVariant action = m_sharedMemory.read(serializableId);
                assert(std::holds_alternative<UnitData>(unit) &&
                       "Read Serializable must be of type Action at this point");

                actions.push_back(std::get<Action>(std::move(action));
            }
        } catch (const std::out_of_range&) {
            /*
             * getLayout() wirft std::out_of_range, wenn serializableId
             * nicht mehr in m_layouts vorhanden ist.
             */
            break;
        }

        ++serializableId;
    }

    return actions;
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

}  // namespace UnBARableAINS::memory
