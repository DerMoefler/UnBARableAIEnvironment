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

template <typename T, typename... Tail>
struct FindType_MF<Typelist<T, Tail...>, T> : std::integral_constant<std::size_t, 0> {};

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

/**
 * \brief Metafunction to rebind a typelist.
 * \tparam Rebind Type to rebind to.
 * \tparam List Typelist to rebind.
 *
 * Produces Rebind<Elements...> for the Elements of the list.
 * Example: Rebind_MF<std::tuple, Typelist<int, float>>::Type = std::tuple<int, float>
 */
template <template <typename...> typename Rebind, typename List>
struct Rebind_MF;

/**
 * \brief Implemenation for Rebind_MF.
 * \tparam Rebind Type to rebind to.
 * \tparam Elements Types to rebind.
 */
template <template <typename...> typename Rebind, typename... Elements>
struct Rebind_MF<Rebind, Typelist<Elements...>> {
    using Type = Rebind<Elements...>;
};

/**
 * \brief Metafunction to rebind a Typelist using a \p Mapper.
 * \tparam Mapper Type to remap the \p List into.
 * \tparam Rebind Type to hold the mappers.
 * \tparam List Typelist to rebind.
 *
 * Produces Rebind<Mapper<Elements>...> for the Elements of the list.
 * Example:
 * \code{.cpp}
 * template <typename T>
 * struct Holder {};
 *
 * TransformRebind_MF<Holder, std::tuple, Typelist<int, float>>::Type
 * = std::tuple<Holder<int>, Holder<float>>
 * \endcode
 */
template <template <typename> typename Mapper, template <typename...> typename Rebind,
          typename List>
struct TransformRebind_MF;

/**
 * \brief Implementation for TransformRebind_MF.
 * \tparam Mapper Type to remap the \p List into.
 * \tparam Rebind Type to hold the mappers.
 * \tparam Elements The Elements of the Typelist.
 */
template <template <typename> typename Mapper, template <typename...> typename Rebind,
          typename... Elements>
struct TransformRebind_MF<Mapper, Rebind, Typelist<Elements...>> {
    using Type = Rebind<Mapper<Elements>...>;
};

namespace test {
using TestList = Typelist<bool, int, std::size_t, float>;

static_assert(FindType_MF<TestList, bool>::value == 0, "Wrong index");
static_assert(FindType_MF<TestList, int>::value == 1, "Wrong index");
static_assert(FindType_MF<TestList, std::size_t>::value == 2, "Wrong index");
static_assert(FindType_MF<TestList, float>::value == 3, "Wrong index");

template <typename... Args>
struct TestRebind {};

static_assert(
    std::same_as<Rebind_MF<TestRebind, TestList>::Type, TestRebind<bool, int, std::size_t, float>>,
    "Failure");

template <typename S>
struct TestHolder {};

static_assert(std::same_as<TransformRebind_MF<TestHolder, TestRebind, TestList>::Type,
                           TestRebind<TestHolder<bool>, TestHolder<int>, TestHolder<std::size_t>,
                                      TestHolder<float>>>);

}  // namespace test

template <typename List, typename T>
struct Contains_MF : std::false_type {};

template <typename T, typename... Elements>
struct Contains_MF<Typelist<Elements...>, T>
    : std::bool_constant<(std::same_as<T, Elements> || ...)> {};

}  // namespace Typelist
