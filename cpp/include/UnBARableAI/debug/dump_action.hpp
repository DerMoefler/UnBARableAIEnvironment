#ifndef DUMP_ACTION_H_
#define DUMP_ACTION_H_

#include <cstddef>
#include <ostream>

#include "UnBARableAI/action.h"

namespace UnBARableAINS {

namespace debug {

void dumpAction(std::ostream& out, const Action& action, std::size_t indentationLevel = 0);

}

}  // namespace UnBARableAINS

#endif  // DUMP_ACTION_H_
