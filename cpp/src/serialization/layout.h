#ifndef LAYOUT_H_
#define LAYOUT_H_

#include <cstddef>
#include <optional>
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
template <Serializable S>
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
    inline static constexpr std::string_view c_debug_name_serializable = debug::typeName<S>();
    inline static constexpr std::string_view c_debug_name_end = ">";

public:
    /// \brief Type alias for \p T's SerializeInformation.
    using SI = SerializeInformation<S>;
    /// \brief Type alias for the std::tuple holding the \ref FieldNode%s.
    using Nodes = typename GetNodesTupleType_MF<typename SI::Fields>::Nodes;

    inline static constexpr std::string_view c_debug_name =
        JoinStringViews_v<c_debug_name_start, c_debug_name_serializable, c_debug_name_end>;

    /// \brief Constant for number of Fields.
    inline static constexpr size_t c_num_fields = std::tuple_size<Nodes>::value;

    inline static constexpr std::size_t c_num_multi_fields =
        detail::GetMultiFieldCount_MF<S>::value;

    /**
     * \todo IMPLEMENT
     */
    Layout(void) { buildNodes(); }

    /**
     * \brief Basic constructor to compute a layout.
     * \param value An instance of Type \p T for which to compute the layout.
     */
    Layout(const S& value) { buildNodes(value); }

    template <typename F>
    void forEachNode(F&& func) {
        std::apply([&](auto&... node) -> void { (func(node), ...); }, m_nodes);
    }

    template <typename F>
    void forEachNode(F&& func) const {
        std::apply([&](const auto&... node) -> void { (func(node), ...); }, m_nodes);
    }

    /**
     * \brief Execute a function \p func for each node's child layout(s).
     * \tparam F Type of \p func.
     * \tparam executeBefore Boolean specyfing whether to execute the function before or after.
     * \param func A function invokable like func(Layout<...>* layout).
     *
     * Executes the function for each node's child layout recursively. The execution for the parent
     * (i.e. the Layout on which you call this method) can happen either before or after the
     * recursion which is specified by \p executeBefore.
     */
    template <typename F, bool executeBefore>
    void visitNodeChildLayoutsRecursively(F&& func) {
        if constexpr (executeBefore) {
            func(this);
        }
        // Recursion
        forEachNode([&](auto& node) {
            visitNodeChildLayouts(node, [&](auto& childLayout) {
                childLayout->visitNodeChildLayoutsRecursively(func);
            });
        });
        if constexpr (!executeBefore) {
            func(this);
        }
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
     * Calls \ref getFieldNode for each \p Nodes... and computes the cumulated sizes.
     */
    void buildNodes(void) {
        size_t offset = 0;
        m_inlinedSize = 0;
        m_deepSize = 0;
        std::apply(
            [&](auto&... fields) {
                ((fields = buildDefaultNode<typename std::remove_cvref_t<decltype(fields)>::Tag>(
                      offset)),
                 ...);
                ((m_inlinedSize += fields.inlineSize), ...);
                ((m_deepSize += fields.deepSize), ...);
            },
            m_nodes);
    }

    /**
     * \brief Helper to build the actual \ref Nodes.
     * \param[in] value An instance of Type \p T.
     * Calls \ref getFieldNode for each \p Nodes... and computes the cumulated sizes.
     */
    void buildNodes(const S& value) {
        size_t offset = 0;
        m_inlinedSize = 0;
        m_deepSize = 0;
        std::apply(
            [&](auto&... fields) mutable {
                ((fields = buildNode<typename std::remove_cvref_t<decltype(fields)>::Tag>(value,
                                                                                          offset)),
                 ...);
                ((m_inlinedSize += fields.inlineSize), ...);
                ((m_deepSize += fields.deepSize), ...);
            },
            m_nodes);
    }

    /**
     * \brief Helper to build one FieldNode for a specific \ref FieldlikeConcept "Field".
     * \tparam F The \ref FieldlikeConcept "Fieldlike".
     * \param[inout] currentOffset The cumulated inlined sizes of the Fields before this Field in
     * the parent data.
     */
    template <detail::Fieldlike F>
    static FieldNode<F> buildDefaultNode(size_t& currentOffset) {
        using ValueType = typename detail::GetFieldlikeValueType_MF<F>::Type;
        using ValueTypeSI = SerializeInformation<ValueType>;

        FieldNode<F> node{};
        node.offset = currentOffset;

        // A MultiField is currently default constructed to have no elements.
        if constexpr (detail::MultiField<F>) {
            node.count = 0;
            node.deepSize = 0;
        }
        else if constexpr (detail::ConstSize<ValueType>) {
            constexpr size_t serializedSize = ValueTypeSI::c_serialized_size;
            node.deepSize = serializedSize;
        }
        else {
            node.child = std::make_shared<Layout<ValueType>>();
            node.deepSize = node.child->getDeepSize();
        }

        initNodeInlineSize(node);

        currentOffset += node.inlineSize;
        return node;
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
    static FieldNode<F> buildNode(const ParentValue& parentValue, size_t& currentOffset) {
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

        initNodeInlineSize(node);

        currentOffset += node.inlineSize;
        return node;
    }

    /**
     * \brief Helper to initialize the node's inline size (might also affect the deep size).
     * \param node Node to update.
     *
     * This function initializes the inline size of a node. If the node is not fully inlined,
     * the required inline size is added to the deep size.
     */
    template <detail::Fieldlike F>
    static void initNodeInlineSize(FieldNode<F>& node) {
        using ValueType = typename detail::GetFieldlikeValueType_MF<F>::Type;

        constexpr bool isTypeInlined = detail::isTypeInlined<ValueType>();
        constexpr bool isFieldInlined = detail::isFieldInlined<F>();

        if constexpr (FieldNode<F>::isFullyInlined) {
            node.inlineSize = node.deepSize;
        }
        // Field itself is inlined, but the type isnt. Imagine an std::vector<std::vector<int>>.
        // The F_Elements MultiField of a vector is generally inlined. The first
        // F_Elements::ValueType is std::vector<int> however, which itself is not inlineable.
        // Therefore, we can only inline the first F_Elements as link_t's to the std::vector<int>.
        else if constexpr (detail::MultiField<F> && isFieldInlined && !isTypeInlined) {
            node.inlineSize = node.count * sizeof(memory::link_t);
            node.deepSize += node.inlineSize;
        }
        // Doesnt matter what kind of field it is, its in another segment both times.
        else {
            node.inlineSize = sizeof(memory::link_t);
            node.deepSize += node.inlineSize;
        }
    }

    size_t m_inlinedSize = 0;
    size_t m_deepSize = 0;
    std::optional<memory::id_t> m_segmentId = std::nullopt;
    Nodes m_nodes;
};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // LAYOUT_H_
