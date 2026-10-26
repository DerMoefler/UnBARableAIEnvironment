#include "memory/segment_table.hpp"

#include <format>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stdexcept>

#include "serialization/serialize_information.h"

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

SegmentTable::SegmentTable(std::vector<PartialSegmentTable> tables)
    : m_partialSegmentTables(std::move(tables)) {
    // TODO validation, m_idAllocator and free chunk building
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
    return m_partialSegmentTables[index];
}

position_t SegmentTable::allocateChunk(const std::size_t chunkSize) {
    // TODO use free chunks here as well instead of default allocating from the head
    position_t position = m_head;
    m_head += chunkSize;
    return position;
}

}  // namespace memory

}  // namespace UnBARableAINS
