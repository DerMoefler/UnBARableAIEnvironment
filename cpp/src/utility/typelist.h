#pragma once
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "utility/always_false.h"

namespace Typelist {

namespace detail {
/** \brief A simple wrapper to use lazy instantiation on expressions that would otherwise be ill
 * formed.
 * \tparam T Type to wrap.
 *
 * std::type_identity does provide the same functionality, but with type instead of Type as the name
 * for the wrapped Type.
 */
template <typename T>
struct Wrapper {
    /// \brief Result,
    using Type = T;
};
}  // namespace detail

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

    using Type = typename std::conditional_t<index == 0, detail::Wrapper<Head>,
                                             TypeAtIndex_MF<TailTypelist, index - 1>>::Type;
};

/// \brief Type alias for \ref TypeAtIndex_MF::Type.
template <typename List, std::size_t index>
using TypeAtIndex_T = typename TypeAtIndex_MF<List, index>::Type;

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
 * \brief Metafunction to get the front of a Typelist.
 * \tparam List Typelist.
 */
template <typename List>
struct Front_MF;

/// \brief Partial specialization for implementation.
template <typename Head, typename... Tail>
struct Front_MF<Typelist<Head, Tail...>> {
    using Type = Head;
};

/// \brief Type alias for \ref Front_MF::Type.
template <typename List>
using Front_T = Front_MF<List>::Type;

/**
 * \brief Metafunction to pop the type at the front of the \p List.
 * \tparam List Current Typelist.
 */
template <typename List>
struct PopFront_MF;

/// \brief Partial specialization for implementation.
template <typename Head, typename... Tail>
struct PopFront_MF<Typelist<Head, Tail...>> {
    using Type = Typelist<Tail...>;
};

/// \brief Type alias for \ref PopFront_MF::Type.
template <typename List>
using PopFront_T = typename PopFront_MF<List>::Type;

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

/// \brief Type alias for PushFront_MF::Type.
template <typename List, typename NewElement>
using PushFront_T = typename PushFront_MF<List, NewElement>::Type;

/**
 * \brief Metafunction to pop the back of the \p List.
 * \tparam List Current Typelist.
 */
template <typename List>
struct PopBack_MF;

/// \brief Recursive case: Keep head, compute PopBack for Tail.
template <typename Head, typename... Tail>
    requires(sizeof...(Tail) > 0)
struct PopBack_MF<Typelist<Head, Tail...>> {
    using Type = PushFront_T<typename PopBack_MF<Typelist<Tail...>>::Type, Head>;
};

/// \brief Base case: Remove last element, i.e. compute empty Typelist.
template <typename Last>
struct PopBack_MF<Typelist<Last>> {
    using Type = Typelist<>;
};

/// \brief Type alias for \ref PopBack_MF::Type.
template <typename List>
using PopBack_T = typename PopBack_MF<List>::Type;

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
 * \brief Metafunction to check that a \p Predicate is true for every Element in the \p List.
 * \tparam List Typelist to apply the predicate to.
 * \tparam Predicate Predicate to verify.
 */
template <typename List, template <typename> typename Predicate>
struct AllOf_MF;

/**
 * \brief Implementation, applies \p Predicate to each Element.
 * \tparam Elements Elements of the Typelist.
 * \tparam Predicate Predicate to verify.
 */
template <typename... Elements, template <typename> typename Predicate>
struct AllOf_MF<Typelist<Elements...>, Predicate>
    : std::bool_constant<(Predicate<Elements>::value && ...)> {};

/// \brief Value template for AllOf_MF.
template <typename List, template <typename> typename Predicate>
static constexpr bool AllOf_v = AllOf_MF<List, Predicate>::value;

/**
 * \brief Metafunction to get a sublist from \p List, starting at \p offset and having \p elements.
 * \tparam List Base Typelist.
 * \tparam offset Offset in \p List.
 * \tparam count Number of types to keep starting at offset.
 *
 * Recursively pops the front of the \p List until offset = 0.
 * \see Size_MF, PopFront_MF, Sublist_MF<List, 0, count>.
 */
template <typename List, std::size_t offset, std::size_t count>
struct Sublist_MF {
    static_assert(offset + count <= Size_MF<List>::value, "Sublist cannot be larger than List");
    static_assert(count > 0, "Count cannot be 0");

    using Sublist = PopFront_T<List>;
    using Type = typename Sublist_MF<Sublist, offset - 1, count>::Type;
};

/**
 * \brief Specialization for offset = 0.
 * \tparam List Base Typelist.
 * \tparam count Number of types to keep.
 *
 * Recursively pop the back of the \p List until its size is equal to \p count.
 * \see Size_MF, PopBack_MF
 */
template <typename List, std::size_t count>
struct Sublist_MF<List, 0, count> {
    static_assert(count <= Size_MF<List>::value, "Sublist cannot be larger than List");
    static_assert(count > 0, "Count cannot be 0");

    using Sublist = PopBack_T<List>;
    using Type = typename std::conditional<Size_MF<List>::value == count, detail::Wrapper<List>,
                                           Sublist_MF<Sublist, 0, count>>::type::Type;
};

/// \brief Type alias for Sublist_MF::Type.
template <typename List, std::size_t offset, std::size_t count>
using Sublist_T = typename Sublist_MF<List, offset, count>::Type;

/**
 * \brief Execute \p Metafun on each element of the \p List, building a new Typelist with the
 * results.
 * \tparam List The Typelist.
 * \tparam Metafun Metafunction to apply each Elements. Computed type must be available via
 * Metafun<Element>::Type.
 */
template <typename List, template <typename> typename Metafun>
struct Transform_MF;

/// \brief Implementation.
template <typename... Elements, template <typename> typename Metafun>
struct Transform_MF<Typelist<Elements...>, Metafun> {
    using Type = Typelist<typename Metafun<Elements>::Type...>;
};

/// \brief Type alias for \ref Transform_MF::Type.
template <typename List, template <typename> typename Metafun>
using Transform_T = typename Transform_MF<List, Metafun>::Type;

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

/// \brief Type aliast for \ref Rebind_MF::Type.
template <template <typename...> typename Rebind, typename List>
using Rebind_T = Rebind_MF<Rebind, List>::Type;

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
 * \todo Refactor or something now that we have Transform_MF.
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

static_assert(Size_MF<TestList>::value == 4, "Incorrect size for Testlist");
static_assert(std::same_as<PopBack_T<TestList>, Typelist<bool, int, std::size_t>>,
              "PopBack does not work");

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

static_assert(std::same_as<Sublist_T<TestList, 0, 2>, Typelist<bool, int>>,
              "Sublists do not match");
static_assert(std::same_as<Sublist_T<TestList, 1, 2>, Typelist<int, std::size_t>>,
              "Sublists do not match");
static_assert(std::same_as<Sublist_T<TestList, 2, 2>, Typelist<std::size_t, float>>,
              "Sublists do not match");

}  // namespace test

}  // namespace Typelist
