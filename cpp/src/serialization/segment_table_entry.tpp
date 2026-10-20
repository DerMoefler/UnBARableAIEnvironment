#ifndef SEGMENT_TABLE_ENTRY_TPP_
#define SEGMENT_TABLE_ENTRY_TPP_

#include <type_traits>

#include "memory/segment_table_entry.hpp"
#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

template <>
struct SerializeInformation<memory::SegmentTableEntry> {
    using Type = memory::SegmentTableEntry;

    static constexpr bool c_serializable = Serializable<std::optional<memory::link_t>>;

    struct F_Id : detail::Field_t {
        using Type = memory::id_t;
    };

    struct F_Link : detail::Field_t {
        using Type = std::optional<memory::link_t>;
    };

    using Fields = detail::Fields<F_Id, F_Link>;

    static constexpr memory::id_t get(std::type_identity<F_Id>, const Type& value) {
        return value.getId();
    }

    static constexpr std::optional<memory::link_t> get(std::type_identity<F_Link>,
                                                       const Type& value) {
        return value.getLink();
    }

    static constexpr Type constructFromFields(const memory::id_t id,
                                              const std::optional<memory::link_t> link) {
        return memory::SegmentTableEntry{id, link};
    }
};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
