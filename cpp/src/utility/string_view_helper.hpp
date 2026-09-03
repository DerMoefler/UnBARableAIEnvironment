#ifndef STRING_VIEW_HELPER_H_
#define STRING_VIEW_HELPER_H_

#include <array>
#include <cstddef>
#include <string_view>

/**
 * \brief Helper metafunction to join std::string_views.
 * \tparam strs String views to concatenate.
 *
 * Implementation origin:
 * https://stackoverflow.com/questions/38955940/how-to-concatenate-static-strings-at-compile-time
 */
template <const std::string_view&... strs>
struct JoinStringViews_MF {
private:
    /**
     * \brief Implementation to allocate an array containing all the characters of the strings.
     * \returns Array of all the strings' characters.
     */
    static constexpr auto impl(void) noexcept {
        constexpr size_t resultLength = (strs.size() + ... + 0);
        std::array<char, resultLength + 1> arr{};
        auto append = [i = 0, &arr](auto const& s) mutable {
            for (auto c : s) arr[i++] = c;
        };
        (append(strs), ...);
        arr[resultLength] = 0;
        return arr;
    }
    /// \brief Static storage in form of an array.
    static constexpr auto arr = impl();

public:
    /// \brief Computed value (view on \ref arr).
    static constexpr std::string_view value{arr.data(), arr.size() - 1};
};

/// \brief Value template helper for \ref JoinStringViews_MF.
template <const std::string_view&... strs>
static constexpr auto JoinStringViews_v = JoinStringViews_MF<strs...>::value;

#endif  // STRING_VIEW_HELPER_H_
