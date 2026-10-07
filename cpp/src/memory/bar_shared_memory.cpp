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
    std::cout << "Writing UnitData for team ID: " << unitData.team_id << std::endl;
    if (!m_ownTeamId.has_value()) {
        m_ownTeamId = unitData.team_id;
    }
    std::cout << "Own team ID cached" << std::endl;
    return m_sharedMemory.write(unitData);
}

BarSharedMemory::SerializableId BarSharedMemory::writeEngineStatus(EngineStatus status) {
    return m_sharedMemory.write(status);
}

int BarSharedMemory::getOwnTeamId() { return resolveOwnTeamId(); }

int BarSharedMemory::resolveOwnTeamId() {
    if (m_ownTeamId.has_value()) {
        return m_ownTeamId.value();
    }

    /*
     * Nach BarSharedMemory::open() wurde über dieses C++-Objekt noch
     * keine UnitData geschrieben. Daher wird die erste gespeicherte
     * UnitData gelesen.
     */
    std::vector<BarSharedMemory::UnitData> units =
        readAll<BarSharedMemory::UnitData>();

    if (units.empty()) {
        throw std::runtime_error(
            "Cannot determine own team ID because no UnitData "
            "exists in shared memory.");
    }

    m_ownTeamId = units.front().team_id;

    return m_ownTeamId.value();
}

}  // namespace UnBARableAINS::memory
