#ifndef TYPE_INDEX_H_
#define TYPE_INDEX_H_

#include <cstddef>
#include <cstdint>

/**
 * \brief Metafunction to get the index of a type into the list.
 * \tparam T Type to find.
 * \tparam List... All the types, to find T in.
 *
 * If multiple T are inside the List, returns the first.
 */
template <typename T, typename... List>
struct TypeIndex_MF;

/// \brief Base case: T is at the front of the list.
template <typename T, typename... Others>
struct TypeIndex_MF<T, T, Others...> {
    inline static constexpr std::size_t index = 0;
};

/// \brief Recursive case: Pop the front of the list and recurse.
template <typename S, typename U, typename... List>
struct TypeIndex_MF<S, U, List...> {
    inline static constexpr std::size_t index = TypeIndex_MF<S, List...>::index + 1;
};

// Simple test
static_assert(TypeIndex_MF<int, uint8_t, uint16_t, int>::index == 2,
              "Int has incorrect index in (uint8_t, uint16_t, int)");

#endif  // TYPE_INDEX_H_
