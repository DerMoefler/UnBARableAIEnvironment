#ifndef PRINT_HELPERS_H_
#define PRINT_HELPERS_H_

#include <cstddef>
#include <ostream>
#include <string_view>

namespace UnBARableAINS {

namespace debug {

inline void makeIndentation(std::ostream& out, std::size_t depth) {
    for (size_t i = 0; i < depth; i++) {
        out << "\t";
    }
}

inline std::string_view getBoolValueDisplayName(bool value) {
    if (value) {
        return "True";
    }
    else {
        return "False";
    }
}

}  // namespace debug

}  // namespace UnBARableAINS

#endif  // PRINT_HELPERS_H_
