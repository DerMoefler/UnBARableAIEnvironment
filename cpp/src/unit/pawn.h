#pragma once
#include "unit.h"
#include "unit_data/pawn_data.h"

namespace UnBARableAI {

namespace unit {

class Pawn : public Unit {
public:
    Pawn(PawnData data);

    float getHealth(void) const override;

private:
    PawnData m_data;

};

}; // namespace unit

}; // namespace UnBARableAI