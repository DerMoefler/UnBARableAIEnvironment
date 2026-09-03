#ifndef TYPE_NAME_H_
#define TYPE_NAME_H_

#include <string_view>

namespace UnBARableAINS {

namespace serialization {

namespace debug {

template <typename T>
constexpr std::string_view typeName() {
#if defined(__clang__)
    constexpr std::string_view function = __PRETTY_FUNCTION__;
    constexpr std::string_view prefix = "T = ";
    constexpr auto begin = function.find(prefix) + prefix.size();
    constexpr auto end = function.rfind(']');

    return function.substr(begin, end - begin);
#elif defined(__GNUC__) || defined(__GNUG__)
    constexpr std::string_view function = __PRETTY_FUNCTION__;
    constexpr std::string_view prefix = "with T = ";
    constexpr auto begin = function.find(prefix) + prefix.size();
    constexpr auto end = function.find(';', begin);

    return function.substr(begin, end - begin);
#else
    return "<unknown type>";
#endif
}

}  // namespace debug

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // TYPE_NAME_H_
