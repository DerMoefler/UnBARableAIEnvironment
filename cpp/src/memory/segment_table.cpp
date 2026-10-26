#include "memory/segment_table.hpp"

#include <cassert>
#include <format>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stdexcept>

#include "serialization/serialize_information.h"
#include "serialization/trivial_serializable.hpp"

#include "serialization/array.tpp"

static auto logger = spdlog::stdout_color_mt("PartialSegmentTable");

namespace UnBARableAINS {

namespace memory {

template <serialization::Serializable S>
using SI = serialization::SerializeInformation<S>;

namespace detail {

PartialSegmentTable::PartialSegmentTable(position_t offset, SerializedView serialized)
    : PartialSegmentTable(offset, getIdAtIndex(serialized, 0)) {}

bool PartialSegmentTable::updateLinks(SerializedView serialized) {
    bool change = false;
    for (std::size_t i = 0; i < c_segments_count; i++) {
        auto& segment = m_segments[i];
        const id_t readId = getIdAtIndex(serialized, i);
        const id_t expectedId = segment.getId();
        if (readId != expectedId) {
            std::string errorMsg = std::format(
                "Serialized data starts with id {}, but partial segment table starts with {}",
                readId, expectedId);
            SPDLOG_LOGGER_ERROR(logger, errorMsg);
            throw std::invalid_argument(errorMsg);
        }

        const link_t link = getLinkAtIndex(serialized, i);
        if (!link) {
            SPDLOG_LOGGER_DEBUG(logger, "Segment with id {} is empty", expectedId);
            if (segment.isValid()) {
                SPDLOG_LOGGER_TRACE(logger, "Segment {} was deleted since last update.",
                                    expectedId);
                segment = segment.invalidate();
                change = true;
            }
        }
        else {
            position_t newMemoryStart = static_cast<link_t>(link);
            if (segment.isValid()) {
                position_t oldMemoryStart = segment.getMemoryStart();
                if (newMemoryStart == oldMemoryStart) {
                    continue;
                }
                else {
                    change = true;
                    segment = SegmentInformation(expectedId, newMemoryStart, 0);
                }
            }
            else {
                change = true;
                segment = SegmentInformation(expectedId, newMemoryStart, 0);
            }
        }
    }

    return change;
}

id_t PartialSegmentTable::deserializeAtPosition(SerializedView serialized, position_t position) {
    auto serializedId = std::span<const std::byte, sizeof(id_t)>{
        serialized.data() + static_cast<std::size_t>(position), sizeof(id_t)};
    return SI<id_t>::deserialize(serializedId);
}

}  // namespace detail

// SegmentTable::SegmentTable(std::vector<PartialSegmentTable> tables)
//     : m_partialSegmentTables(std::move(tables)) {
//     // TODO validation, m_idAllocator and free chunk building
// }

const SegmentInformation& SegmentTable::allocate(const std::size_t segmentDataSize) {
    const std::size_t totalSegmentSize = SegmentInformation::c_header_size + segmentDataSize;

    const position_t offset = allocateChunk(totalSegmentSize);
    const auto id = static_cast<memory::id_t>(m_idAllocator.allocate());
    assert(m_segments[id] == std::nullopt && "SegmentInformation at id exists already");
    m_segments[id] = SegmentInformation{id, offset, segmentDataSize};
    assert(m_segments[id].value().totalSize() == totalSegmentSize &&
           "Wrong totalSegmentSize calculation used.");

    m_partialSegmentTables[getTableIndex(id)].updateEntryLink(getIntableIndex(id),
                                                              static_cast<link_t>(offset));

    return m_segments[id].value();
}

bool SegmentTable::isFull(void) const {
    for (const auto& table : m_partialSegmentTables) {
        if (!table.isFull()) {
            return false;
        }
    }
    return true;
}

const PartialSegmentTable& SegmentTable::extend(void) {
    const std::size_t index = m_partialSegmentTables.size();
    position_t offset = allocateChunk(c_partial_table_size);
    m_partialSegmentTables.push_back(PartialSegmentTable(offset, index));
    m_segments.insert(m_segments.end(), PartialSegmentTable::c_segments_count, std::nullopt);
    return m_partialSegmentTables[index];
}

const SegmentInformation& SegmentTable::getSegmentInformation(const id_t segmentId) const {
    throwOnInvalidId(segmentId);
    return m_segments[segmentId].value();
}

const position_t SegmentTable::getSegmentEntryPosition(const id_t segmentId) const {
    if (segmentId >= m_segments.size()) throw std::out_of_range("SegmentId out of range");

    using F_Entries = typename serialization::SerializeInformation<PartialSegmentTable>::F_Entries;
    using ArraySI = serialization::SerializeInformation<typename F_Entries::Type>;

    constexpr std::size_t entriesFieldOffset = serialization::getFieldOffset(
        std::type_identity<PartialSegmentTable>{}, std::type_identity<F_Entries>{});
    constexpr std::size_t entrySize = ArraySI::c_element_size;

    const position_t segmentTableOffset =
        m_partialSegmentTables[getTableIndex(segmentId)].getOffset();
    const std::size_t intableIndex = getIntableIndex(segmentId);

    return segmentTableOffset +
           static_cast<position_t>(entriesFieldOffset + entrySize * intableIndex);
}

void SegmentTable::advanceSegmentInformationHead(const id_t segmentId,
                                                 const std::size_t increment) {
    throwOnInvalidId(segmentId);
    m_segments[segmentId].value().advanceHead(increment);
}

position_t SegmentTable::allocateChunk(const std::size_t chunkSize) {
    // TODO use free chunks here as well instead of default allocating from the head
    position_t position = m_head;
    m_head += chunkSize;
    return position;
}

void SegmentTable::throwOnInvalidId(const id_t segmentId) const {
    if (segmentId >= m_segments.size()) throw std::out_of_range("SegmentId out of range");
    if (!m_segments[segmentId].has_value()) throw std::invalid_argument("Segment uninitialized.");
}

}  // namespace memory

}  // namespace UnBARableAINS
