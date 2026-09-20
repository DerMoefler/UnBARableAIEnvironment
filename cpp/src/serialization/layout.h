#ifndef LAYOUT_H_
#define LAYOUT_H_

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

#include "serialize_information.h"
#include "layout_update_context.hpp"
#include "layout_value_recursion_context.hpp"
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

    Layout(void) { buildNodes(); }

    template <template <typename> typename Context>
        requires LayoutUpdateContext<Context, S>
    Layout(const Context<S>& context)
        : Layout() {
        update(context);
    }

    /**
     * \brief Basic constructor to compute a layout.
     * \param value An instance of Type \p T for which to compute the layout.
     */
    Layout(const S& value)
        : Layout() {
        update(value);
    }

    /**
     * \brief Equality operator.
     * \param other Layout to compare.
     */
    bool operator==(const Layout<S>& other) const {
        if (m_inlinedSize != other.m_inlinedSize || m_deepSize != other.m_deepSize) {
            return false;
        }

        // Lambda to compare the child layout (std::shared_ptrs atm).
        auto compareChildLayouts = [&](const auto& lhsChild, const auto& rhsChild) -> bool {
            if (lhsChild == nullptr || rhsChild == nullptr) {
                return lhsChild == rhsChild;
            }
            return *lhsChild == *rhsChild;
        };

        // Comparison of two nodes
        auto compare = [&](const auto& lhsNode, const auto& rhsNode) -> bool {
            using Tag = typename std::remove_cvref_t<decltype(lhsNode)>::Tag;
            using RhsTag = typename std::remove_cvref_t<decltype(rhsNode)>::Tag;
            static_assert(std::same_as<Tag, RhsTag>, "Node Tags must be exactly the same.");

            if (lhsNode.inlineSize != rhsNode.inlineSize || lhsNode.deepSize != rhsNode.deepSize) {
                return false;
            }

            if constexpr (detail::MultiField<Tag>) {
                if (lhsNode.children.size() != rhsNode.children.size()) {
                    return false;
                }
                for (std::size_t i = 0; i < lhsNode.children.size(); i++) {
                    if (!compareChildLayouts(lhsNode.children[i], rhsNode.children[i])) {
                        return false;
                    }
                }
                return true;
            }
            else {
                return compareChildLayouts(lhsNode.child, rhsNode.child);
            }
        };

        // Immediately invoked lambda with a fold expression to call comapre for each node pair.
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) -> bool {
            return (compare(std::get<Is>(m_nodes), std::get<Is>(other.m_nodes)) && ...);
        }(std::make_index_sequence<std::tuple_size_v<Nodes>>{});
    }

    inline std::size_t update(const S& value) {
        return update(LayoutValueRecursionContext{this, value});
    }

    /**
     * \brief Update a Layout based off a function \p countFunc.
     * \tparam F Type of the count function.
     * \param countFunc A function invocable like countFunc(Layout* layout,
     * std::type_identity<Fieldlike F>{}, std::size_t& offset).
     *
     * This method works in two stages. First, it rebuilds the layout top down. This essentially
     * boils down to each MultiField Node having to (possibly) create/delete childLayouts. This is
     * what \p countFunc is required for.
     *
     * The second stage is then to sum up the sizes bottom up.
     */
    template <template <typename> typename Context>
        requires LayoutUpdateContext<Context, S>
    std::size_t update(const Context<S>& context) {
        size_t offset = 0;
        m_inlinedSize = 0;
        m_deepSize = 0;

        // Reconstructs a MultiFieldNode based off the countFunc.
        auto reconstructMultiFieldNode = [&](auto& node) mutable -> void {
            using Node = std::remove_cvref_t<decltype(node)>;
            using Tag = typename Node::Tag;
            using ValueType = typename Node::ValueType;

            std::size_t newCount = context.template getCount<Tag>();
            node.count = newCount;
            node.deepSize = 0;
            node.inlineSize = 0;

            if constexpr (detail::ConstSize<ValueType>) {
                constexpr std::size_t serializedSize =
                    SerializeInformation<ValueType>::c_serialized_size;
                node.children.resize(0);
                node.deepSize = node.count * serializedSize;
            }
            else {
                node.children.resize(newCount);
                for (std::size_t i = 0; i < newCount; i++) {
                    auto& child = node.children[i];
                    if (!child) {
                        child = std::make_shared<Layout<ValueType>>();
                    }
                    child->update(context.descend(child.get(), node, i));
                    node.deepSize += child->getDeepSize();
                }
            }
            initNodeInlineSize(node);
        };

        forEachNode([&](auto& node) mutable -> void {
            using Tag = typename std::remove_cvref_t<decltype(node)>::Tag;
            node.offset = offset;

            if constexpr (detail::MultiField<Tag>) {
                reconstructMultiFieldNode(node);
            }
            else {
                if (node.child) {
                    node.child->update(context.descend(node.child.get(), node));
                    node.deepSize = node.child->getDeepSize();
                    initNodeInlineSize(node);
                }
            }

            offset += node.inlineSize;
            m_inlinedSize += node.inlineSize;
            m_deepSize += node.deepSize;
        });

        return m_inlinedSize;
    }

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
    template <bool executeBefore, typename F>
    void visitNodeChildLayoutsRecursively(F&& func) {
        if constexpr (executeBefore) {
            func(this);
        }
        // Recursion
        forEachNode([&](auto& node) {
            visitNodeChildLayouts(node, [&](auto& childLayout) {
                childLayout->template visitNodeChildLayoutsRecursively<executeBefore>(func);
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
