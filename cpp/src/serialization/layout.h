#ifndef LAYOUT_H_
#define LAYOUT_H_

#include <cstddef>
#include <string_view>
#include <type_traits>

#include "serialize_information.h"
#include "debug/type_name.hpp"
#include "memory/shared_memory_types.h"
#include "utility/string_view_helper.hpp"

namespace UnBARableAINS {

namespace serialization {

/**
 * \brief Holds information about the memory layout for a Serializable Type.
 * \tparam T A \ref Serializable.
 */
template <Serializable T>
class Layout {
private:
    /**
     * \brief Metafunction to get the Type of a std::tuple to hold all FieldNode%s.
     * \tparam Fields \ref Fields.
     */
    template <typename Fields>
    struct GetNodesTupleType_MF;

    /**
     * \brief Metafunction to get the Type of a std::tuple to hold all FieldNode%s.
     * \tparam Fields \ref Fields.
     * Partial specialization provides the actual implementation.
     */
    template <detail::Fieldlike... Fs>
    struct GetNodesTupleType_MF<detail::Fields<Fs...>> {
        using Nodes = std::tuple<FieldNode<Fs>...>;
    };

    inline static constexpr std::string_view c_debug_name_start = "Layout<";
    inline static constexpr std::string_view c_debug_name_serializable = debug::typeName<T>();
    inline static constexpr std::string_view c_debug_name_end = ">";

public:
    /// \brief Type alias for \p T's SerializeInformation.
    using SI = SerializeInformation<T>;
    /// \brief Type alias for the std::tuple holding the \ref FieldNode%s.
    using Nodes = typename GetNodesTupleType_MF<typename SI::Fields>::Nodes;

    inline static constexpr std::string_view c_debug_name =
        JoinStringViews_v<c_debug_name_start, c_debug_name_serializable, c_debug_name_end>;

    /// \brief Constant for number of Fields.
    inline static constexpr size_t c_num_fields = std::tuple_size<Nodes>::value;

    /**
     * \brief Basic constructor to compute a layout.
     * \param value An instance of Type \p T for which to compute the layout.
     */
    Layout(const T& value) { buildNodes(value, m_nodes, m_inlinedSize, m_deepSize); }

    template <typename F>
    void forEachNode(F&& func) {
        std::apply([&](auto&... node) -> void { (func(node), ...); }, m_nodes);
    }

    template <typename F>
    void forEachNode(F&& func) const {
        std::apply([&](const auto&... node) -> void { (func(node), ...); }, m_nodes);
    }

    /**
     * \brief A generic function to execute functions on the nodes based on whether they're inlined.
     * \param funcInlined Function to execute when the node/its children is/are inlined.
     * \param funcNonInlined Function to execute otherwise.
     *
     * This function allows to easily execute functions on nodes. Specifically, it enables writing
     * easy recursion:
     ```cpp
     template <Serializable S>
     void foo(Layout<S> layout) {
         auto funcBase = [&](auto& node) { ... Stuff to do when inlined ... };
         auto funcRecursive = [&](auto& node) { foo(child); };

         layout.visitNodesByInlining(funcBase, funcRecursive);
     }
     ```
     */
    template <typename Func_Inlined, typename Func_NonInlined>
    void visitNodesByInlining(Func_Inlined&& funcInlined, Func_NonInlined&& funcNonInlined) {
        auto func = [&](auto& node) -> void {
            if constexpr (std::remove_cvref_t<decltype(node)>::isFullyInlined) {
                // --- Node entirely inlined ---
                funcInlined(node);
            }
            else {
                funcNonInlined(node);
            }
        };
        forEachNode(func);
    }

    /**
     * \brief A generic function to execute functions on every nodes' child layouts, based on
     * whether the child is inlined.
     * \param funcInlined A function taking in (parentNode, [shared_ptr] childLayout), executed when
     * child is inlined.
     * \param funcNoninlined A function taking in (parentNode, [shared_ptr] childLayout), executed
     * when child is not inlined.
     *
     * This method uses \ref visitNodesByInlining to visit the children based on their inlining.
     */
    template <typename Func_Inlined, typename Func_NonInlined>
    void visitChildLayoutsByInlining(Func_Inlined&& funcInlined, Func_NonInlined&& funcNonInlined) {
        auto parentFuncInlined = [&](auto& parentNode) {
            visitNodeChildLayouts(parentNode,
                                  [&](auto& childLayout) { funcInlined(parentNode, childLayout); });
        };
        auto parentFuncNonInlined = [&](auto& parentNode) {
            visitNodeChildLayouts(parentNode, [&](auto& childLayout) {
                if (childLayout->isInlined()) {
                    funcInlined(parentNode, childLayout);
                }
                else {
                    funcNonInlined(parentNode, childLayout);
                }
            });
        };

        visitNodesByInlining(parentFuncInlined, parentFuncNonInlined);
    }

    inline bool isInlined(void) const { return m_inlinedSize == m_deepSize; }

    /**
     * \brief Getter for inlinedSize.
     * \returns Inlined size for \p T.
     */
    inline size_t getInlinedSize(void) const { return m_inlinedSize; };

    /**
     * \brief Getter for deepSize, i.e. the entire size for the serialization for the given value of
     * \p T in memory.
     * \returns Deep size  for \p T.
     */
    inline size_t getDeepSize(void) const { return m_deepSize; };

    /**
     * \brief Getter for (optional) segmentId.
     * \returns (optional) segmentId.
     */
    inline std::optional<memory::id_t> getSegmentId(void) const { return m_segmentId; }

    /**
     * \brief Setter for (optional) segmentId.
     * \param segmentId New segmentId.
     */
    inline void setSegmentId(std::optional<memory::id_t> segmentId) { m_segmentId = segmentId; }

    /**
     * \brief Getter for all nodes.
     * \returns Tuple of all nodes.
     */
    inline decltype(auto) getNodes(void) { return m_nodes; }

    /**
     * \brief Getter for a specific \ref FieldNode.
     * \tparam F The requested \ref FieldlikeConcept "Field".
     * \returns The corresponding FieldNode.
     */
    template <detail::Fieldlike F>
    inline const FieldNode<F>& get(std::type_identity<F>) const {
        return std::get<FieldNode<F>>(m_nodes);
    };

private:
    /**
     * \brief Helper to build the actual \ref Nodes.
     * \tparam Nodes The types of \ref Nodes.
     * \param[in] value An instance of Type \p T.
     * \param[inout] tuple A std::tuple<Nodes...> holding the actual data.
     * \param[out] inlinedSize The inlined size for the given \p value.
     * \param[out] deepSize The deep size for the given \p value.
     * Calls \ref getFieldNode for each \p Nodes... and computes the cumulated sizes.
     */
    template <typename... Nodes>
    static void buildNodes(const T& value, std::tuple<Nodes...>& tuple, size_t& inlinedSize,
                           size_t& deepSize) {
        size_t offset = 0;
        std::apply(
            [&](Nodes&... fields) {
                ((fields = getFieldNode<typename Nodes::Tag>(value, offset)), ...);
                ((inlinedSize += fields.inlineSize), ...);
                ((deepSize += fields.deepSize), ...);
            },
            tuple);
    }

    /**
     * \brief Helper to build one FieldNode for a specific \ref FieldlikeConcept "Field".
     * \tparam F The \ref FieldlikeConcept "Fieldlike".
     * \tparam ParentValue The Serializable Type the Field is a part of.
     * \param[in] parentValue An instance of Type \p ParentValue.
     * \param[inout] currentOffset The cumulated inlined sizes of the Fields before this Field in
     * the parent data.
     */
    template <detail::Fieldlike F, Serializable ParentValue>
    static FieldNode<F> getFieldNode(const ParentValue& parentValue, size_t& currentOffset) {
        // Alias and constants
        using ValueType = typename detail::GetFieldlikeValueType_MF<F>::Type;
        using ValueTypeSI = SerializeInformation<ValueType>;
        using ParentValueSI = SerializeInformation<ParentValue>;
        constexpr std::type_identity<F> fieldKey{};

        FieldNode<F> node{};
        node.offset = currentOffset;

        // if constexpr (inlineField<F>() && !inlineType<ValueType>()) {
        //     static_assert(AlwaysFalse_MF<F>::value, "The type of the field specified to be
        //     inlined is not actually inlineable.");
        // }

        // Get number of elements for MultiField
        if constexpr (detail::MultiField<F>) {
            node.count = ParentValueSI::getSize(fieldKey, parentValue);
        }

        // ----- Compute the deep size of the field. -----
        // End of recursion: The ValueType has a constant size and the Field's size is therefore
        // easily computed.
        if constexpr (detail::ConstSize<ValueType>) {
            constexpr size_t serializedSize = ValueTypeSI::c_serialized_size;
            if constexpr (detail::MultiField<F>) {
                node.deepSize = node.count * serializedSize;
            }
            else {
                node.deepSize = serializedSize;
            }
        }
        // Recursive case: A Layout has to be computed for the ValueType since it is more complex
        // than a constant size.
        else {
            // Compute a layout for each element of the multifield.
            if constexpr (detail::MultiField<F>) {
                node.deepSize = 0;
                // TODO does this work as intended?
                node.children.reserve(node.count);
                for (int i = 0; i < node.count; i++) {
                    decltype(auto) fieldValue = ParentValueSI::get(fieldKey, parentValue, i);
                    node.children.push_back(std::make_shared<Layout<ValueType>>(fieldValue));
                    node.deepSize += node.children[i]->getDeepSize();
                }
            }
            // Compute a layout for the one element we have in a simple field.
            else {
                decltype(auto) fieldValue = ParentValueSI::get(fieldKey, parentValue);
                node.child = std::make_shared<Layout<ValueType>>(fieldValue);
                node.deepSize = node.child->getDeepSize();
            }
        }
        // ----- Deep size computed. -----

        // ----- Update inline size. -----
        constexpr bool isTypeInlined = detail::inlineType<ValueType>();
        constexpr bool isFieldInlined = detail::inlineField<F>();
        //  Fully inlined
        if constexpr (FieldNode<F>::isFullyInlined) {
            node.inlineSize = node.deepSize;
        }
        // Field itself is inlined, but the type isnt. Imagine an std::vector<std::vector<int>>.
        // The F_Elements MultiField of a vector is generally inlined. The first
        // F_Elements::ValueType is std::vector<int> however, which itself is not inlineable.
        // Therefore, we can only inline the first F_Elements as link_t's to the std::vector<int>.
        else if constexpr (detail::MultiField<F> && isFieldInlined && !isTypeInlined) {
            node.inlineSize = node.count * sizeof(memory::link_t);
        }
        else {
            node.inlineSize = sizeof(memory::link_t);
        }
        // ----- Inline size updated. -----

        currentOffset += node.inlineSize;
        return node;
    }

    size_t m_inlinedSize = 0;
    size_t m_deepSize = 0;
    std::optional<memory::id_t> m_segmentId = std::nullopt;
    Nodes m_nodes;
};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // LAYOUT_H_
