#ifndef SEGMENT_TABLE_HEADER_TPP_
#define SEGMENT_TABLE_HEADER_TPP_

#include <cstddef>
#include <type_traits>

#include "memory/segment_table_header.hpp"
#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

template <>
struct SerializeInformation<memory::SegmentTableHeader> {
    using Type = memory::SegmentTableHeader;

    static constexpr bool c_serializable = Serializable<std::optional<memory::link_t>>;

    struct F_NumElements : detail::Field_t {
        using Type = std::size_t;
    };

    struct F_Link : detail::Field_t {
        using Type = std::optional<memory::link_t>;
    };

    using Fields = detail::Fields<F_NumElements, F_Link>;

    static constexpr std::size_t get(std::type_identity<F_NumElements>, const Type value) {
        return Type::c_contiguous_count;
    }

    static constexpr std::optional<memory::link_t> get(std::type_identity<F_Link>,
                                                       const Type value) {
        return value.getLink();
    }

    static constexpr Type constructFromFields(const std::size_t numElements,
                                              const std::optional<memory::link_t> link) {
        return Type{numElements, link};
    }
};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
