#include <gtest/gtest.h>

#include "UnBARableAI/bar_shared_memory.h"
#include "UnBARableAI/action.h"
#include "UnBARableAI/engine_status.h"
#include "UnBARableAI/grpc_connection.hpp"
#include "UnBARableAI/unit_data.h"

namespace UnBARableAINS {

using namespace memory;

TEST(BarSharedMemoryTest, UpdateEngineSession) {
    constexpr std::string_view name = "/shm-bar-update-engine-session";
    constexpr EngineStatus status = EngineStatus::GAME_ENDED;
    constexpr GrpcConnection connection{50000};
    constexpr std::size_t numAllyUnits = 3;
    constexpr std::size_t numEnemyUnits = 3;

    auto shm = BarSharedMemory::create(name, status, connection, numAllyUnits, numEnemyUnits);
    EXPECT_EQ(status, shm.getEngineStatus());
    shm.updateEngineStatus(EngineStatus::RUNNING);
    EXPECT_EQ(EngineStatus::RUNNING, shm.getEngineStatus());

    BarSharedMemory::remove(name);
}

TEST(BarSharedMemoryTest, InitialObservation) {
    constexpr std::string_view name = "/shm-bar-initial-observation";
    constexpr EngineStatus status = EngineStatus::RUNNING;
    constexpr GrpcConnection connection{50000};
    constexpr std::size_t numAllyUnits = 3;

    std::vector<unit::UnitData> units{
        unit::UnitData{1, 2, 3, 4, 100.0f, 120.0f, 0.0f, 0.0f, 0.0f, 75.0f, 5.0f, false, false,
                       100.0f, 0.0f, 0.0f},
        unit::UnitData{2, 2, 3, 4, 80.0f, 120.0f, 17.0f, 18.0f, 19.0f, 75.0f, 200.0f, false, false,
                       100.0f, 0.0f, 0.0f},
        unit::UnitData{3, 2, 3, 4, 50.0f, 120.0f, 0.0f, 0.0f, 0.0f, 75.0f, 5.0f, true, false,
                       100.0f, 56.0f, 0.0f},
    };

    std::vector<Action> actions{Action{0, 1, 2, ActionId::MoveLeft, 3},
                                Action{0, 1, 2, ActionId::MoveDown, 3},
                                Action{0, 1, 2, ActionId::Attack, 3}};

    auto createShm = BarSharedMemory::create(name, status, connection, units.size(), units.size());

    auto openShm = BarSharedMemory::open(name);

    // ----- Units -----
    auto unitIds = openShm.getIdsForType(std::type_identity<BarSharedMemory::UnitDataOpt>{});
    EXPECT_EQ(unitIds.size(), 2 * units.size());

    // Update ally units (enemy units stay std::nullopt)
    for (std::size_t i = 0; i < units.size(); i++) {
        createShm.updateUnitData(unitIds[i], units[i]);
    }
    EXPECT_TRUE(openShm.update());
    // Read ally and enemy units
    // Expect ally units = units and enemy units all = std::nullopt
    auto readUnits = openShm.readAll<BarSharedMemory::UnitDataOpt>();
    EXPECT_EQ(readUnits.size(), 2 * units.size());
    for (std::size_t i = 0; i < units.size(); i++) {
        EXPECT_EQ(readUnits[i], units[i]);
        EXPECT_EQ(readUnits[units.size() + i], std::nullopt);
    }

    // ----- Actions -----
    auto actionIds = openShm.getIdsForType(std::type_identity<BarSharedMemory::ActionOpt>{});
    EXPECT_EQ(actionIds.size(), units.size());
    // Update actions
    for (std::size_t i = 0; i < units.size(); i++) {
        createShm.updateAction(actionIds[i], actions[i]);
    }
    EXPECT_TRUE(openShm.update());
    // Read actions and expect readActions = actions
    auto readActions = openShm.readAll<BarSharedMemory::ActionOpt>();
    EXPECT_EQ(readActions.size(), actions.size());
    for (std::size_t i = 0; i < units.size(); i++) {
        EXPECT_EQ(readActions[i], actions[i]);
    }

    BarSharedMemory::remove(name);
}

}  // namespace UnBARableAINS
