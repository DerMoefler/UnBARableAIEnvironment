#ifndef LAYOUT_UPDATE_CONTEXT_H_
#define LAYOUT_UPDATE_CONTEXT_H_

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

namespace detail {

/**
 * \brief Implementation for \ref LayoutUpdateContext.
 * \tparam Context Type to check.
 * \tparam S The Serializable the Context is for.
 * \tparam F The Field to check support for.
 *
 * Essentially checks that descending is possible (with or without index for MultiField/Field
 * respectively) for the specified \ref Field.
 */
template <template <typename> typename Context, Serializable S, detail::Fieldlike F>
consteval bool doesContextSupportField(void) {
    using ValueType = typename FieldNode<F>::ValueType;

    if constexpr (detail::MultiField<F>) {
        return requires(const Context<S>& context, Layout<ValueType>* childLayout,
                        const FieldNode<F>& node, std::size_t index) {
            { context.template getCount<F>() } -> std::convertible_to<std::size_t>;
            { context.descend(childLayout, node, index) } -> std::same_as<Context<ValueType>>;
        };
    }
    else {
        return requires(const Context<S>& context, Layout<ValueType>* childLayout,
                        const FieldNode<F>& node) {
            { context.descend(childLayout, node) } -> std::same_as<Context<ValueType>>;
        };
    }
}

/// \brief Metafunction to check \ref doesContextSupportField for a Typelist.
template <template <typename> typename Context, Serializable S, typename FieldList>
struct ContextSupportsFields_MF;

/// \brief Implementation for Metafunction to check \ref doesContextSupportField for a Typelist.
template <template <typename> typename Context, Serializable S, detail::Fieldlike... Fs>
struct ContextSupportsFields_MF<Context, S, detail::Fields<Fs...>>
    : std::bool_constant<(doesContextSupportField<Context, S, Fs>() && ...)> {};

}  // namespace detail

/**
 * \brief Concept to check whether some type \p Context can be used inside Layout::update
 * \tparam Context Type to check.
 * \tparam S The same Serializable as the Layout update is called for.
 *
 * Checks that Context supplies a descend methods that can be called like descend(childLayout, node)
 * or descend(childLayout, node, index) for Fields/Multifields respectively.
 *
 * For Multifields, the \p Context must also provide a Context::getCount<NodeTag>(), which is the
 * actual functionality we care about in the Layout::update method.
 *
 */
template <template <typename> typename Context, typename S>
concept LayoutUpdateContext =
    Serializable<S> &&
    detail::ContextSupportsFields_MF<Context, S, typename SerializeInformation<S>::Fields>::value;

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // LAYOUT_UPDATE_CONTEXT_H_
