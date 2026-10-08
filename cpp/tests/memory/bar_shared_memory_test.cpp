#include <gtest/gtest.h>

#include "UnBARableAI/bar_shared_memory.h"
#include "UnBARableAI/action.h"
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

    auto createShm = BarSharedMemory::create("/shm-BAR-constructor");

    for (const auto& unit : units) {
        createShm.writeUnitData(unit);
    }

    for (const auto& action : actions) {
        createShm.writeAction(action);
    }

    auto openShm = BarSharedMemory::open("/shm-BAR-constructor");
    auto readUnits = openShm.readAll<unit::UnitData>();
    auto readActions = openShm.readAll<Action>();

    ASSERT_TRUE(units.size() == readUnits.size());

    for (std::size_t i = 0; i < units.size(); i++) {
        EXPECT_EQ(units[i], readUnits[i]);
    }

    ASSERT_TRUE(actions.size() == readActions.size());
    for (std::size_t i = 0; i < actions.size(); i++) {
        EXPECT_EQ(actions[i], readActions[i]);
    }

    BarSharedMemory::remove("/shm-BAR-constructor");
}

}  // namespace UnBARableAINS
