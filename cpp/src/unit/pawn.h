#pragma once
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

private:
    UnitData m_data;
};

}; // namespace unit

}; // namespace UnBARableAI