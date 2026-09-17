#ifndef LAYOUT_DUMP_H_
#define LAYOUT_DUMP_H_

#include <cstddef>
#include <optional>
#include <ostream>
#include <type_traits>

#include "serialization/debug/type_name.hpp"
#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

namespace debug {

namespace detail {

inline void makeIndentation(std::ostream& out, size_t depth) {
    for (size_t i = 0; i < depth; i++) {
        out << "\t";
    }
}

}  // namespace detail

template <serialization::detail::Fieldlike Tag>
void dumpNode(std::ostream& out, const FieldNode<Tag>& node, size_t depth) {
    using Node = std::remove_cvref_t<decltype(node)>;

    detail::makeIndentation(out, depth);
    out << displayName<Tag>() << '\n';
}

template <Serializable S>
void dumpLayout(std::ostream& out, const Layout<S>& layout, size_t depth) {
    using Layout = Layout<S>;

    auto printOpt = [&out]<typename T>(const std::optional<T> opt) {
        if (opt.has_value()) {
            out << opt.value();
        }
        else {
            out << "std::nullopt";
        }
    };

    detail::makeIndentation(out, depth);
    out << displayName<Layout>() << '\n';
    detail::makeIndentation(out, depth + 1);
    out << "SegmentID: ";
    printOpt(layout.getSegmentId());
    out << "\n";
    detail::makeIndentation(out, depth + 1);
    out << "Inline size: " << layout.getInlinedSize() << '\n';
    detail::makeIndentation(out, depth + 1);
    out << "Deep size: " << layout.getDeepSize() << '\n';

    layout.forEachNode([&](const auto& node) {
        dumpNode(out, node, depth + 1);
        visitNodeChildLayouts(
            node, [&](const auto& childLayout) { dumpLayout(out, *childLayout, depth + 2); });
    });
}

}  // namespace debug

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // LAYOUT_DUMP_H_
