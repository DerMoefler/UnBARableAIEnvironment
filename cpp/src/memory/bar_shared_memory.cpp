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

BarSharedMemory BarSharedMemory::create(const std::string_view name, const EngineStatus status,
                                        const GrpcConnection& connection,
                                        const std::size_t initialNumAllyUnits,
                                        const std::size_t initialNumEnemyUnits) {
    const std::size_t numSerializables = 2 + initialNumAllyUnits * 2 + initialNumEnemyUnits;
    auto impl = Impl::create(name, numSerializables);
    // 0. Engine Status
    const auto engineStatusId = impl.write<EngineStatus>(status);
    throwOnInvalidId(c_engine_status_id, engineStatusId, "EngineStatus");
    // 1 .Connection
    const auto connectionId = impl.write<GrpcConnection>(connection);
    throwOnInvalidId(c_grpc_connection_id, connectionId, "GrpcConnection");
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
    SPDLOG_LOGGER_INFO(logger, "Created BarSharedMemory \"{}\" for an (initial) {} vs {} scenario.",
                       name, initialNumAllyUnits, initialNumEnemyUnits);
    return BarSharedMemory{std::move(impl)};
}

BarSharedMemory BarSharedMemory::open(const std::string_view name) {
    BarSharedMemory result{Impl::open(name)};

    auto throwOnSerializablesCountMismtach = []<typename T, typename Allocator>(
                                                 const std::vector<T, Allocator>& vector,
                                                 const std::string_view name) -> void {
        if (vector.size() != 1) {
            const std::string message = std::format(
                "There are {} {} instances, but only one is expected.", vector.size(), name);
            SPDLOG_LOGGER_ERROR(logger, message);
            throw std::runtime_error(message);
        }
    };

    std::vector<SerializableId> statusIds{};
    const auto statuses = result.readAll<EngineStatus>(&statusIds);
    throwOnSerializablesCountMismtach(statuses, "EngineStatus");
    throwOnInvalidId(c_engine_status_id, statusIds.at(0), "EngineStatus");

    std::vector<SerializableId> connectionIds{};
    const auto connections = result.readAll<GrpcConnection>(&connectionIds);
    throwOnSerializablesCountMismtach(connections, "GrpcConnection");
    throwOnInvalidId(c_grpc_connection_id, connectionIds.at(0), "GrpcConnection");

    SPDLOG_LOGGER_INFO(logger, "Opened BarSharedMemory \"{}\"", name);
    return result;
}

void BarSharedMemory::remove(std::string_view name) { Impl::remove(std::string{name}); }

BarSharedMemory::SerializableId BarSharedMemory::writeAction(const Action& action) {
    return m_sharedMemory.write<ActionOpt>(action);
}

BarSharedMemory::SerializableId BarSharedMemory::writeUnitData(const UnitData& unitData) {
    /*
     * Die erste über diesen Wrapper geschriebene UnitData bestimmt
     * die eigene Team-ID.
     */
    if (!m_ownTeamId.has_value()) {
        m_ownTeamId = unitData.team_id;
    }
    return m_sharedMemory.write<UnitDataOpt>(unitData);
}

void BarSharedMemory::updateEngineStatus(const EngineStatus status) {
    m_sharedMemory.updateSerializable(c_engine_status_id, status);
}

EngineStatus BarSharedMemory::getEngineStatus(void) const {
    return std::get<EngineStatus>(m_sharedMemory.read(c_engine_status_id));
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
            "Cannot determine own team ID because no UnitDataOpts exist in shared memory.");
    }

    for (const auto& unit : units) {
        if (unit.has_value()) {
            m_ownTeamId = unit.value().team_id;
            return m_ownTeamId.value();
        }
    }

    throw std::runtime_error("Cannot determine own team ID because all UnitDataOpts are empty.");
}

void BarSharedMemory::throwOnInvalidId(const SerializableId expectedId,
                                       const SerializableId actualId, const std::string_view name) {
    if (expectedId != actualId) {
        const std::string message =
            std::format("ID {} for {} does not match expected ID {}", actualId, name, actualId);
        SPDLOG_LOGGER_ERROR(logger, message);
        throw std::runtime_error(message);
    }
}

}  // namespace UnBARableAINS::memory
