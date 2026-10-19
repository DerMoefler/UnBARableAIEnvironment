#ifndef SEGMENT_HEADER_TPP_
#define SEGMENT_HEADER_TPP_

#include <type_traits>
#include "memory/segment_header.hpp"
#include "memory/shared_memory_types.h"
#include "serialization/optional.tpp"
#include "serialization/serialize_information.h"
#include "serialization/trivial_serializable.hpp"

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
static_assert(TrivialSerializable<memory::SegmentHeader> &&
              "SegmentHeader is not trivially serializable.");
static_assert(getSerializableSize(std::type_identity<memory::SegmentHeader>{}) == 17,
              "Incorrect size for SegmentHeader (or implemenation changed, in that case just "
              "change the comparison value).");
static_assert(getFieldOffset(
                  std::type_identity<memory::SegmentHeader>{},
                  std::type_identity<SerializeInformation<memory::SegmentHeader>::F_TotalSize>{}) ==
                  0,
              "Wrong offset for F_TotalSize (or implementation changed).");
static_assert(
    getFieldOffset(std::type_identity<memory::SegmentHeader>{},
                   std::type_identity<SerializeInformation<memory::SegmentHeader>::F_Head>{}) == 8,
    "Wrong offset for F_Head (or implementation changed).");
static_assert(
    getFieldOffset(
        std::type_identity<memory::SegmentHeader>{},
        std::type_identity<SerializeInformation<memory::SegmentHeader>::F_NextSegmentId>{}) == 12,
    "Wrong offset for F_TotalSize (or implementation changed).");

constexpr memory::SegmentHeader header{16};

constexpr auto serializedHeader = serialize(header);
constexpr auto deserializedHeader = deserialize<memory::SegmentHeader>(serializedHeader);

constexpr memory::SegmentHeader headerWithLink{
    getSerializableSize(std::type_identity<memory::SegmentHeader>{}) + 16, 7, 1};

constexpr auto serializedWithLink = serialize(headerWithLink);
constexpr auto deserializedWithLink = deserialize<memory::SegmentHeader>(serializedWithLink);

// TODO fix
// static_assert(deserializedWithLink.remainingSize() == 9, "Reamining size is wrong");

}  // namespace test

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
