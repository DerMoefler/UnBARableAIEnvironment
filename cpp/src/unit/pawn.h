#pragma once
#include <vector>
#include "unit.h"
#include "../unit_data/unit_data.h"

namespace UnBARableAINS {

namespace unit {

class Pawn : public Unit {
public:
    Pawn(UnitData data);
    float getMaxHealth(void) const override {
        return 370.0f;
    }
    float getSightRange(void) const {
        return 429.0f;
    }
};

}; // namespace unit

}; // namespace UnBARableAI