#pragma once
#include <vector>
#include "unit.h"
#include "../unit_data/unit_data.h"

namespace UnBARableAI {

namespace unit {

class Pawn : public Unit {
public:
    Pawn(UnitData data);

    float getHealth(void) const override {
        return m_data.health;
    }
    int getTeam(void) const override {
        return m_data.team;
    }
    float getXPosition(void) const override {
        return m_data.xPosition;
    }
    float getYPosition(void) const override {
        return m_data.yPosition;
    }
    float getZPosition(void) const override {
        return m_data.zPosition;
    }
    bool hasCurrentCommand(void) const override {
        return m_data.hasCurrentCommand;
    }

    int getUnitID(void) const override {
        return m_data.unitID;
    }
    int getUnitType(void) const override {
        return m_data.unitType;
    }
    std::vector<int> getUnitsInSight(void) const override {
        return std::vector<int>{1, 2, 3};
    }
    std::vector<int> getEnemyUnitsInSight(void) const override {
        return std::vector<int>{4, 5, 6};
    }

private:
    UnitData m_data;
};

}; // namespace unit

}; // namespace UnBARableAI