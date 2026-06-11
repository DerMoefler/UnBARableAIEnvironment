#pragma once
#include "unit_data/unit_data.h"
#include <vector>

namespace UnBARableAINS {

class EngineBridge {
public:
    EngineBridge() = default;

    void writeUnitDataInMemory(unit::UnitData data);
    void writeUnitsInSightInMemory(int unitId, std::vector<int> unitsInSight);
    void writeEnemyUnitsInSightInMemory(int unitId, std::vector<int> enemyUnitsInSight);

};

}; // namespace UnBARableAI
