#pragma once
#include <vector>
#include "../unit_data/unit_data.h"

namespace UnBARableAINS {

namespace unit {

class Unit {
public:
    Unit(UnitData data) : m_data(data) {}

    float getHealth(void) const {
        return m_data.health;
    }

    virtual float getMaxHealth(void) const = 0;
    virtual float getSightRange(void) const = 0;

    int getTeam(void) const {
        return m_data.team;
    }
    float getXPosition(void) const {
        return m_data.xPosition;
    }
    float getYPosition(void) const {
        return m_data.yPosition;
    }
    float getZPosition(void) const {
        return m_data.zPosition;
    }
    bool hasCurrentCommand(void) const {
        return m_data.hasCurrentCommand;
    }
    int getUnitID(void) const {
        return m_data.unitID;
    }
    int getUnitType(void) const {
        return m_data.unitType;
    }
    std::vector<int> getUnitsInSight(void) const {
        return std::vector<int>{1, 2, 3};
    }
    std::vector<int> getEnemyUnitsInSight(void) const {
        return std::vector<int>{4, 5, 6};
    }


private:
    UnitData m_data;
};

}; // namespace unit

}; // namespace UnBARableAI