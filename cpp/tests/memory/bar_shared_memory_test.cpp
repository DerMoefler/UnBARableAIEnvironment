#include <gtest/gtest.h>

#include "UnBARableAI/bar_shared_memory.h"
#include "UnBARableAI/action.h"
#include "UnBARableAI/engine_status.h"
#include "UnBARableAI/grpc_connection.hpp"
#include "UnBARableAI/unit_data.h"

namespace UnBARableAINS {

using namespace memory;

TEST(BarSharedMemoryTest, Constructor) {
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

    EngineStatus status = EngineStatus::RUNNING;
    GrpcConnection connection{50000};

    auto createShm = BarSharedMemory::create("/shm-BAR-constructor", status, connection,
                                             units.size(), units.size());
    EXPECT_EQ(status, createShm.getEngineStatus());
    createShm.updateEngineStatus(EngineStatus::AI_CRASHED);
    EXPECT_EQ(EngineStatus::AI_CRASHED, createShm.getEngineStatus());

    BarSharedMemory::remove("/shm-BAR-constructor");
}

}  // namespace UnBARableAINS
