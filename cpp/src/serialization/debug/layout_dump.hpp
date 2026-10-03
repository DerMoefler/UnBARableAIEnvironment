#ifndef LAYOUT_DUMP_H_
#define LAYOUT_DUMP_H_

#include <cstddef>
#include <optional>
#include <ostream>
#include <type_traits>

#include "serialization/debug/type_name.hpp"
#include "serialization/serialize_information.h"
#include "utility/debug/print_helpers.hpp"

namespace UnBARableAINS {

namespace serialization {

namespace debug {

template <serialization::detail::Fieldlike Tag>
void dumpNode(std::ostream& out, const FieldNode<Tag>& node, size_t depth) {
    using Node = std::remove_cvref_t<decltype(node)>;

    UnBARableAINS::debug::makeIndentation(out, depth);
    out << displayName<Tag>() << '\n';

    UnBARableAINS::debug::makeIndentation(out, depth);
    out << "Offset " << node.offset << "\n";

    UnBARableAINS::debug::makeIndentation(out, depth);
    out << "InlineSize " << node.inlineSize << "\n";

    UnBARableAINS::debug::makeIndentation(out, depth);
    out << "DeepSize " << node.deepSize << "\n";
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

    UnBARableAINS::debug::makeIndentation(out, depth);
    out << displayName<Layout>() << '\n';
    UnBARableAINS::debug::makeIndentation(out, depth);
    out << "SegmentID: ";
    printOpt(layout.getSegmentId());
    out << "\n";
    UnBARableAINS::debug::makeIndentation(out, depth);
    out << "Inline size: " << layout.getInlinedSize() << '\n';
    UnBARableAINS::debug::makeIndentation(out, depth);
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
