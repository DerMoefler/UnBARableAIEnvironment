#ifndef SEGMENT_HEADER_TPP_
#define SEGMENT_HEADER_TPP_

#include "memory/segment_header.hpp"
#include "memory/shared_memory_types.h"
#include "serialization/optional.tpp"
#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

template <>
struct SerializeInformation<memory::SegmentHeader> {
    using Type = memory::SegmentHeader;

    inline static constexpr bool c_serializable = Serializable<std::optional<memory::id_t>>;

    struct F_TotalSize : detail::Field_t {
        using Type = std::size_t;
    };

    struct F_Head : detail::Field_t {
        using Type = memory::position_t;
    };

    struct F_NextSegmentId : detail::Field_t {
        using Type = std::optional<id_t>;
    };

    using Fields = detail::Fields<F_TotalSize, F_Head, F_NextSegmentId>;

    static constexpr std::size_t get(std::type_identity<F_TotalSize>, const Type& value) {
        return value.getTotalSize();
    }

    static constexpr memory::position_t get(std::type_identity<F_Head>, const Type& value) {
        return value.getHead();
    }

    static constexpr std::optional<memory::id_t> get(std::type_identity<F_NextSegmentId>,
                                                     const Type& value) {
        return value.getNextSegmentId();
    }

    static constexpr Type constructFromFields(const std::size_t totalSize,
                                              const memory::position_t head,
                                              const std::optional<memory::id_t> nextSegmentId) {
        return memory::SegmentHeader{totalSize, head, nextSegmentId};
    };
};

namespace test {

static_assert(Serializable<memory::SegmentHeader>, "SegmentHeader is not serializable.");

}

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
