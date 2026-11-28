#include "../../include/UnBARableAI/bar_shared_memory.h"

#include <format>
#include <magic_enum/magic_enum.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "UnBARableAI/grpc_connection.hpp"

static auto logger = spdlog::stdout_color_mt("BarSharedMemory");

namespace UnBARableAINS::memory {

BarSharedMemory::BarSharedMemory(Impl impl)
    : m_sharedMemory(std::move(impl)) {}

BarSharedMemory BarSharedMemory::create(const std::string_view name, const EngineStatus status,
                                        const GrpcConnection& connection,
                                        const std::size_t initialNumAllyUnits,
                                        const std::size_t initialNumEnemyUnits) {
    const std::size_t numSerializables = 1 + initialNumAllyUnits * 2 + initialNumEnemyUnits;
    auto impl = Impl::create(name, numSerializables);
    // 0. Engine Status
    const EngineInformation information = {
        .engineStatus = status, .grpcConnection = connection, .isTruncated = false};
    const auto engineInformationId = impl.write<EngineInformation>(information);
    throwOnInvalidId(c_engine_information_id, engineInformationId, "EngineInformation");
    // 1. Ally units
    for (std::size_t i = 0; i < initialNumAllyUnits; i++) {
        impl.write<UnitDataOpt>(std::nullopt);
    }
    // 2. Enemy units
    for (std::size_t i = 0; i < initialNumEnemyUnits; i++) {
        impl.write<UnitDataOpt>(std::nullopt);
    }
    // 3. Actions
    for (std::size_t i = 0; i < initialNumAllyUnits; i++) {
        impl.write<ActionOpt>(std::nullopt);
    }
    SPDLOG_LOGGER_INFO(logger, "Created BarSharedMemory \"{}\" for an (initial) {} vs {} scenario.",
                       name, initialNumAllyUnits, initialNumEnemyUnits);
    return BarSharedMemory{std::move(impl)};
}

BarSharedMemory BarSharedMemory::open(const std::string_view name) {
    BarSharedMemory result{Impl::open(name)};
    SPDLOG_LOGGER_DEBUG(logger, "Shared-memory implementation opened successfully");

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

    std::vector<SerializableId> informationIds{};
    const auto informations = result.readAll<EngineInformation>(&informationIds);
    throwOnSerializablesCountMismtach(informations, "EngineInformation");
    throwOnInvalidId(c_engine_information_id, informationIds.at(0), "EngineStatus");

    SPDLOG_LOGGER_INFO(logger, "Opened BarSharedMemory \"{}\"", name);
    return result;
}

void BarSharedMemory::remove(std::string_view name) { Impl::remove(std::string{name}); }

void BarSharedMemory::updateAction(const SerializableId actionId, const ActionOpt& action) {
    m_sharedMemory.updateSerializable(actionId, action);
}

void BarSharedMemory::updateUnitData(const SerializableId unitId, const UnitDataOpt& unitData) {
    if (!m_ownTeamId.has_value() && unitData.has_value()) {
        m_ownTeamId = unitData.value().team_id;
    }

    m_sharedMemory.updateSerializable(unitId, unitData);
}

void BarSharedMemory::clearUnitsAndActions(void) {
    auto clearType = [&]<serialization::Serializable S>(const std::type_identity<S> id) -> void {
        const std::vector<SerializableId> ids = getIdsForType(id);
        for (const auto id : ids) {
            m_sharedMemory.updateSerializable<S>(id, std::nullopt);
        }
    };

    clearType(std::type_identity<ActionOpt>{});
    clearType(std::type_identity<UnitDataOpt>{});
}

auto BarSharedMemory::getEmptyUnitSlot(const std::set<SerializableId>& excludeIds) const
    -> std::optional<SerializableId> {
    const auto unitIds = getIdsForType(std::type_identity<UnitDataOpt>{});
    for (const auto id : unitIds) {
        if (excludeIds.contains(id)) {
            continue;
        }
        auto unit = std::get<UnitDataOpt>(m_sharedMemory.read(id));
        if (!unit.has_value()) {
            return id;
        }
    }
    return std::nullopt;
}

void BarSharedMemory::updateEngineStatus(const EngineStatus status) {
    auto information = getEngineInformation();
    if (information.engineStatus == status) {
        SPDLOG_LOGGER_DEBUG(logger, "Status is already set to {}, skipping write.",
                            magic_enum::enum_name(status));
        return;
    }
    information.engineStatus = status;
    m_sharedMemory.updateSerializable(c_engine_information_id, information);
}

EngineStatus BarSharedMemory::getEngineStatus(void) const {
    const auto information = getEngineInformation();
    return information.engineStatus;
}

GrpcConnection BarSharedMemory::getGrpcConnection(void) const {
    const auto information = getEngineInformation();
    return information.grpcConnection;
}

void BarSharedMemory::setTruncated(const bool isTruncated) {
    auto information = getEngineInformation();
    if (information.isTruncated == isTruncated) {
        SPDLOG_LOGGER_DEBUG(logger, "isTruncated is already set to {}, skipping write.",
                            isTruncated);
        return;
    }
    information.isTruncated = isTruncated;
    m_sharedMemory.updateSerializable(c_engine_information_id, information);
}

bool BarSharedMemory::isTruncated(void) const {
    const auto information = getEngineInformation();
    return information.isTruncated;
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

EngineInformation BarSharedMemory::getEngineInformation(void) const {
    const auto result = m_sharedMemory.read(c_engine_information_id);
    SPDLOG_LOGGER_TRACE(logger, "Read EngineInformation from shared-memory");
    return std::get<EngineInformation>(result);
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
