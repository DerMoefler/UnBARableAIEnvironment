#pragma once

namespace UnBARableAI {

namespace unit {

class Unit {
public:
    virtual float getHealth(void) const = 0;
    virtual int getTeam(void) const = 0;
    virtual float getXPosition(void) const = 0;
    virtual float getYPosition(void) const = 0;
    virtual float getZPosition(void) const = 0;
    virtual bool hasCurrentCommand(void) const = 0;

};

}; // namespace unit

}; // namespace UnBARableAI