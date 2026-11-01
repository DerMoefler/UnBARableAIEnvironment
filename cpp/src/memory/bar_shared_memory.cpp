#include "../../include/UnBARableAI/bar_shared_memory.h"

#include <format>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

static auto logger = spdlog::stdout_color_mt("BarSharedMemory");

namespace UnBARableAINS::memory {

BarSharedMemory::BarSharedMemory(Impl impl)
    : m_sharedMemory(std::move(impl)) {}

BarSharedMemory BarSharedMemory::create(std::string_view name, const EngineStatus status,
                                        const GrpcConnection& connection,
                                        const std::size_t initialNumAllyUnits,
                                        const std::size_t initialNumEnemyUnits) {
    const std::size_t numSerializables = 2 + initialNumAllyUnits * 2 + initialNumEnemyUnits;
    auto impl = Impl::create(name, numSerializables);
    // 0. Engine Status
    const auto engineStatusId = impl.write<EngineStatus>(status);
    if (engineStatusId != c_engine_status_id) {
        const std::string message =
            std::format("ID {} for EngineStatus does not match expected ID {}", engineStatusId,
                        c_engine_status_id);
        SPDLOG_LOGGER_ERROR(logger, message);
        throw std::runtime_error(message);
    }
    // 1 .Connection
    const auto connectionId = impl.write<GrpcConnection>(connection);
    if (connectionId != c_grpc_connection_id) {
        const std::string message =
            std::format("ID {} for GrpcConnection does not match expected ID {}", connectionId,
                        c_grpc_connection_id);
        SPDLOG_LOGGER_ERROR(logger, message);
        throw std::runtime_error(message);
    }
    // 2. Ally units
    for (std::size_t i = 0; i < initialNumAllyUnits; i++) {
        impl.write<UnitDataOpt>(std::nullopt);
    }
    // 3. Enemy units
    for (std::size_t i = 0; i < initialNumEnemyUnits; i++) {
        impl.write<UnitDataOpt>(std::nullopt);
    }
    // 4. Actions
    for (std::size_t i = 0; i < initialNumAllyUnits; i++) {
        impl.write<ActionOpt>(std::nullopt);
    }
    return BarSharedMemory{std::move(impl)};
}

BarSharedMemory BarSharedMemory::open(std::string_view name) {
    return BarSharedMemory{Impl::open(name)};
}

void BarSharedMemory::remove(std::string_view name) { Impl::remove(std::string{name}); }

BarSharedMemory::SerializableId BarSharedMemory::writeAction(const ActionOpt& action) {
    return m_sharedMemory.write(action);
}

BarSharedMemory::SerializableId BarSharedMemory::writeUnitData(const UnitDataOpt& unitData) {
    /*
     * Die erste über diesen Wrapper geschriebene UnitData bestimmt
     * die eigene Team-ID.
     */
    if (!m_ownTeamId.has_value()) {
        m_ownTeamId = unitData.team_id;
    }
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
    std::vector<BarSharedMemory::UnitDataOpt> units = readAll<BarSharedMemory::UnitDataOpt>();

    if (units.empty()) {
        throw std::runtime_error(
            "Cannot determine own team ID because no UnitData "
            "exists in shared memory.");
    }

    m_ownTeamId = units.front().team_id;

    return m_ownTeamId.value();
}

}  // namespace UnBARableAINS::memory
