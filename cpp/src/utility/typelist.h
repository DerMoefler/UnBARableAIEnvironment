#pragma once
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "utility/always_false.h"

namespace Typelist {

template<typename... Elements>
struct Typelist;

template<typename List, typename T>
struct FindType_MF;

template<typename T, typename Head, typename... Tail>
struct FindType_MF<Typelist<Head, Tail...>, T> 
    :   std::integral_constant<size_t, 
            std::same_as<T, Head> ? 0 : 1 + FindType_MF<T, Typelist<Tail...>>::value
        >
{};

// This is only tried to be instantiated when FindType_MF fails. Since no definition is
// given, this will cause a compilation error, telling you the type is not in the list.
template<typename T>
struct FindType_MF<T, Typelist<>> {
    static_assert(AlwaysFalse_MF<T>::value, "This specialization being instantiated means "
        "that the requsted Type does not exist in the given Typelist.");
};

template<typename List, typename T>
struct Contains_MF : std::false_type {};

template<typename T, typename... Elements>
struct Contains_MF<Typelist<Elements...>, T>
    : std::bool_constant<(std::same_as<T, Elements> || ...)> {};

} // namespace typelist