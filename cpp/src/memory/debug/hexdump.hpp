#ifndef HEXDUMP_H_
#define HEXDUMP_H_

#include <istream>
#include <ostream>

namespace UnBARableAINS {

namespace memory {

namespace debug {

void hexdump(std::ostream& out, std::istream& in);

}  // namespace debug

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // HEXDUMP_H_
