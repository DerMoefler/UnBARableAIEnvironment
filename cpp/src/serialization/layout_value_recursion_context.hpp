#ifndef RECURSION_CONTEXT_H_
#define RECURSION_CONTEXT_H_

#include <cstddef>
#include <type_traits>

#include "serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

/**
 * \brief Implementation for a \ref LayoutUpdateContext using the value of a Serializable.
 * \tparam The same Serializable as in Layout<S>::update.
 *
 * Uses the value and \ref SerializeInformation which exposes a getSize for each MultiField.
 * Uses the value obtained from get(node, value[, index]) in descend.
 */
template <Serializable S>
class LayoutValueRecursionContext {
public:
    /// \todo Change to not having to copy value here.
    LayoutValueRecursionContext(const Layout<S>* layout, const S& value)
        : m_layout(layout)
        , m_value(value) {
        assert(m_layout);
    }

    template <detail::MultiField F>
    inline std::size_t getCount(void) const {
        return SerializeInformation<S>::getSize(std::type_identity<F>{}, m_value);
    }

    template <Serializable T, detail::Fieldlike F>
        requires(detail::Field<F>)
    inline auto descend(Layout<T>* childLayout, const FieldNode<F>& node) const
        -> LayoutValueRecursionContext<typename FieldNode<F>::ValueType> {
        return descendImpl(childLayout, node, 0);
    }

    template <Serializable T, detail::Fieldlike F>
        requires(detail::MultiField<F>)
    inline auto descend(Layout<T>* childLayout, const FieldNode<F>& node, std::size_t index) const
        -> LayoutValueRecursionContext<typename FieldNode<F>::ValueType> {
        return descendImpl(childLayout, node, index);
    }

private:
    template <Serializable T, detail::Fieldlike F>
    auto descendImpl(Layout<T>* childLayout, const FieldNode<F>& node, std::size_t index) const
        -> LayoutValueRecursionContext<typename FieldNode<F>::ValueType> {
        using ValueType = typename std::remove_cvref_t<decltype(node)>::ValueType;
        static_assert(std::same_as<ValueType, T>,
                      "Child Layout's ValueType does not match node's ValueType");
        constexpr std::type_identity<F> key{};
        if constexpr (detail::Field<F>) {
            return LayoutValueRecursionContext<ValueType>{
                childLayout, SerializeInformation<S>::get(key, m_value)};
        }
        else {
            return LayoutValueRecursionContext<ValueType>{
                childLayout, SerializeInformation<S>::get(key, m_value, index)};
        }
    }

    const Layout<S>* m_layout;
    S m_value;
};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // RECURSION_CONTEXT_H_
