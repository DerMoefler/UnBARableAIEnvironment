#ifndef TYPE_NAME_H_
#define TYPE_NAME_H_

#include <concepts>
#include <string_view>

namespace UnBARableAINS {

namespace serialization {

namespace debug {

namespace detail {

/**
 * \brief Concept to check whether a type \p T specifies a debug name.
 * \tparam T Type to check.
 *
 * Checks whether the given type \p T provides a static member "c_debug_name" that's convertible to
 * a std::string_view.
 */
template <typename T>
concept HasDebugName = requires {
    { T::c_debug_name } -> std::convertible_to<std::string_view>;
};

}  // namespace detail

/**
 * \brief Helper function to get a printable name for a type \p T.
 * \tparam T to get the name for.
 * \returns std::string_view of the type's name.
 */
template <typename T>
constexpr std::string_view typeName(void) {
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

/**
 * \brief Helper function to get a display name for a type \p T.
 * \tparam T Type to get the name for.
 * \returns std::string_view of the type's display name.
 *
 * Checks whether the type provides a debug name. Otherwise uses \ref typeName.
 */
template <typename T>
constexpr std::string_view displayName(void) {
    if constexpr (detail::HasDebugName<T>) {
        return T::c_debug_name;
    }
    else {
        return typeName<T>();
    }
}

}  // namespace debug

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // TYPE_NAME_H_
