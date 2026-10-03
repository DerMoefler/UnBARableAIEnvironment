#ifndef DUMP_UNIT_DATA_H_
#define DUMP_UNIT_DATA_H_

#include <cstddef>
#include <ostream>

#include "UnBARableAI/unit_data.h"

namespace UnBARableAINS {

namespace debug {

void dumpUnitData(std::ostream& out, const unit::UnitData& unitData,
                  std::size_t indentationLevel = 0);

}

}  // namespace UnBARableAINS

#endif  // DUMP_UNIT_DATA_H_
