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
 * \brief Metafunction to check if a Typelist is empty.
 * \tparam List Typelist to check.
 */
template <typename List>
struct IsEmpty_MF;

/**
 * \brief Metafunction implementation.
 * \tparam Elements Elements of the Typelist.
 *
 * Simply uses the sizeof... operator on \p Elements.
 */
template <typename... Elements>
struct IsEmpty_MF<Typelist<Elements...>>
    : std::bool_constant<!(static_cast<bool>(sizeof...(Elements)))> {};

/**
 * \brief Metafunction to get the size of a Typelist.
 * \taparam List The Typelist.
 */
template <typename List>
struct Size_MF;

/**
 * \brief Metafunction implementation.
 * \tparam Elements Elemenets of the Typelist.
 *
 * Simply uses the sizeof... operator on \p Elements.
 */
template <typename... Elements>
struct Size_MF<Typelist<Elements...>> : std::integral_constant<std::size_t, sizeof...(Elements)> {};

/**
 * \brief Metafuction to get Type of a Typelist at the specified \p index.
 * \tparam List The Typelist.
 * \tparam index Index of the type.
 *
 * Will fail if index > Size_MF<List>::value
 */
template <typename List, std::size_t index>
struct TypeAtIndex_MF;

/**
 * \brief Metafunction implemenentation.
 * \tparam Head The first element of the Typelist.
 * \tparam Tail The remaining Types (if any).
 * \tparam index Index of the searched type.
 *
 * Works recursively by either using Type = Head if index == 0 or calling itself with
 * Typelist<Tail...> and index - 1.
 */
template <typename Head, typename... Tail, std::size_t index>
struct TypeAtIndex_MF<Typelist<Head, Tail...>, index> {
    using List = Typelist<Head, Tail...>;
    static_assert(Size_MF<List>::value > index,
                  "Index cannot be equal to or larger than Typelist size");
    using TailTypelist = Typelist<Tail...>;

    struct HeadWrapper {
        using Type = Head;
    };

    using Type = typename std::conditional_t<index == 0, HeadWrapper,
                                             TypeAtIndex_MF<TailTypelist, index - 1>>::Type;
};

/**
 * \brief Metafunction to check if a Typelist \p List contains a Type \p T.
 * \tparam List The Typelist.
 * \tparam T Type to search for.
 */
template <typename List, typename T>
struct Contains_MF : std::false_type {};

/**
 * \brief Metafunction implementation.
 * \tparam T Type to serach for.
 * \tparam Elements Elements of the Typelist.
 */
template <typename T, typename... Elements>
struct Contains_MF<Typelist<Elements...>, T>
    : std::bool_constant<(std::same_as<T, Elements> || ...)> {};

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

// IsEmpty_MF
static_assert(IsEmpty_MF<Typelist<>>::value, "Typelist<> must be empty");
static_assert(!IsEmpty_MF<TestList>::value, "TestList must be non empty");

// Index_MF
static_assert(std::same_as<typename TypeAtIndex_MF<TestList, 0>::Type, bool>,
              "First (index = 0) element of TestList not equal to bool");
static_assert(std::same_as<typename TypeAtIndex_MF<TestList, 2>::Type, std::size_t>,
              "Third (index = 2) element of TestList not equal to std::size_t");

// FindType_MF
static_assert(FindType_MF<TestList, bool>::value == 0, "Wrong index");
static_assert(FindType_MF<TestList, int>::value == 1, "Wrong index");
static_assert(FindType_MF<TestList, std::size_t>::value == 2, "Wrong index");
static_assert(FindType_MF<TestList, float>::value == 3, "Wrong index");

template <typename... Args>
struct TestRebind {};

// Rebind_MF
static_assert(
    std::same_as<Rebind_MF<TestRebind, TestList>::Type, TestRebind<bool, int, std::size_t, float>>,
    "Failure");

template <typename S>
struct TestHolder {};

// TransformRebind_MF
static_assert(std::same_as<TransformRebind_MF<TestHolder, TestRebind, TestList>::Type,
                           TestRebind<TestHolder<bool>, TestHolder<int>, TestHolder<std::size_t>,
                                      TestHolder<float>>>);

}  // namespace test

}  // namespace Typelist
