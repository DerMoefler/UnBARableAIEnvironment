#ifndef PARTIAL_SEGMENT_TABLE_TPP_
#define PARTIAL_SEGMENT_TABLE_TPP_

#include <type_traits>
#include "memory/partial_segment_table.hpp"
#include "memory/segment_table_header.hpp"
#include "serialization/serialize_information.h"

#include "serialization/array.tpp"
#include "serialization/segment_table_header.tpp"

namespace UnBARableAINS {

namespace serialization {

template <>
struct SerializeInformation<memory::PartialSegmentTable> {
    using Type = memory::PartialSegmentTable;

    static constexpr bool c_serializable = Serializable<memory::SegmentTableHeader> &&
                                           Serializable<memory::PartialSegmentTable::Entries>;

    struct F_Header : detail::Field_t {
        using Type = memory::SegmentTableHeader;
    };

    struct F_Entries : detail::Field_t {
        using Type = memory::PartialSegmentTable::Entries;
    };

    using Fields = detail::Fields<F_Header, F_Entries>;

    static constexpr memory::SegmentTableHeader get(std::type_identity<F_Header>,
                                                    const Type& value) {
        return memory::SegmentTableHeader(memory::PartialSegmentTable::c_segments_count,
                                          value.nextPartialTablePosition());
    }

    static constexpr const memory::PartialSegmentTable::Entries& get(std::type_identity<F_Entries>,
                                                                     const Type& value) {
        return value.getEntries();
    }

    static constexpr Type constructFromFields(memory::SegmentTableHeader header,
                                              const memory::PartialSegmentTable::Entries& entries) {
        return Type{header, entries};
    }
};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
