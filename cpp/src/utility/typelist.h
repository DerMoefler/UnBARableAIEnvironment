#pragma once
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "utility/always_false.h"

namespace Typelist {

/**
 * \brief A simple Typelist.
 * \tparam Elements Type the list contains.
 */
template <typename... Elements>
struct Typelist;

/**
 * \brief Metafunction to push a type to the front a Typelist.
 * \tparam List Current Typelist.
 * \tparam NewElement Element to push to the front.
 * \returns Typelist<NewElement, ListElements...>
 */
template <typename List, typename NewElement>
struct PushFront_MF;

/// \brief Partial specialization for implementation.
template <typename... Elements, typename NewElement>
struct PushFront_MF<Typelist<Elements...>, NewElement> {
    using Type = Typelist<NewElement, Elements...>;
};

template <typename List, typename T>
struct FindType_MF;

template <typename T, typename Head, typename... Tail>
struct FindType_MF<Typelist<Head, Tail...>, T>
    : std::integral_constant<
          size_t, std::same_as<T, Head> ? 0 : 1 + FindType_MF<Typelist<Tail...>, T>::value> {};

// This is only tried to be instantiated when FindType_MF fails. Since no definition is
// given, this will cause a compilation error, telling you the type is not in the list.
template <typename T>
struct FindType_MF<Typelist<>, T> {
    static_assert(AlwaysFalse_MF<T>::value,
                  "This specialization being instantiated means "
                  "that the requsted Type does not exist in the given Typelist.");
};

template <typename List, typename T>
struct Contains_MF : std::false_type {};

template <typename T, typename... Elements>
struct Contains_MF<Typelist<Elements...>, T>
    : std::bool_constant<(std::same_as<T, Elements> || ...)> {};

}  // namespace Typelist
