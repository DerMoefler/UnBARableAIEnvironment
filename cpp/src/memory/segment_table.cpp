#include "memory/segment_table.hpp"

#include <cassert>
#include <format>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stdexcept>

#include "id/id_allocator.hpp"
#include "memory/partial_segment_table.hpp"
#include "serialization/serialize_information.h"
#include "serialization/trivial_serializable.hpp"

#include "serialization/array.tpp"

static auto logger = spdlog::stdout_color_mt("SegmentTable");

namespace UnBARableAINS {

namespace memory {

template <serialization::Serializable S>
using SI = serialization::SerializeInformation<S>;

SegmentTable::SegmentTable(std::vector<PartialSegmentTable> tables,
                           std::vector<std::optional<SegmentHeader>> headers)
    : m_partialSegmentTables(std::move(tables)) {
    // TODO validation, m_idAllocator and free chunk building
    const std::size_t expectedHeaderCount =
        m_partialSegmentTables.size() * PartialSegmentTable::c_segments_count;
    if (headers.size() != expectedHeaderCount) {
        std::string message = std::format(
            "Header's size must be tables.size() * PartialSegmentTable::c_segments_count, i.e. {} "
            "* {}. However, received number of headers is {}",
            m_partialSegmentTables.size(), PartialSegmentTable::c_segments_count, headers.size());
        SPDLOG_LOGGER_ERROR(logger, message);
        throw std::invalid_argument(message);
    }

    m_segments.reserve(expectedHeaderCount);
    for (id_t id = 0; id < expectedHeaderCount; id++) {
        const auto& table = m_partialSegmentTables[getTableIndex(id)];
        const auto& entry = table.getEntries()[getIntableIndex(id)];
        const auto link = entry.getLink();
        const auto& header = headers[id];

        if (link.has_value() != header.has_value()) {
            throw std::runtime_error(std::format("Mismatch between tables and headers."));
        }

        if (link.has_value()) {
            m_segments.push_back(SegmentInformation{id, link.value(), header.value()});
        }
        else {
            m_segments.emplace_back(std::nullopt);
        }
    }
    updateHead();
    updateIdAllocator();
}

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

void SegmentTable::release(const id_t segmentId) {
    // TODO simple implementation, need head update and free chunk building etc.
    throwOnInvalidId(segmentId);

    assert(m_segments[segmentId].value().getId() == segmentId &&
           "Class invariant: Index into m_segments must be equal to segment.getId() at that index");

    m_segments[segmentId] = std::nullopt;
    m_partialSegmentTables[getTableIndex(segmentId)].updateEntryLink(getIntableIndex(segmentId),
                                                                     std::nullopt);
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

void SegmentTable::advanceSegmentInformationHead(const id_t segmentId,
                                                 const std::size_t increment) {
    throwOnInvalidId(segmentId);
    m_segments[segmentId].value().advanceHead(increment);
}

std::size_t SegmentTable::validSegmentsCount(void) const {
    std::size_t count {0};
    for (const auto& table : m_partialSegmentTables) {
        count += table.validSegmentsCount();
    }
    return count;
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

position_t SegmentTable::allocateChunk(const std::size_t chunkSize) {
    // TODO use free chunks here as well instead of default allocating from the head
    position_t position = m_head;
    m_head += chunkSize;
    return position;
}

bool SegmentTable::updateHead(void) {
    // Minimum head position
    const auto oldHead = m_head;
    m_head = static_cast<position_t>(
        serialization::getSerializableSize(std::type_identity<ShmHeader>{}));

    for (const auto& table : m_partialSegmentTables) {
        m_head =
            std::max(m_head, static_cast<position_t>(table.getOffset() + c_partial_table_size));
    }

    for (const auto& segment : m_segments) {
        if (!segment.has_value()) continue;
        const position_t offset = segment.value().getMemoryStart();
        const position_t segmentEnd = offset + static_cast<position_t>(segment.value().totalSize());
        m_head = std::max(m_head, segmentEnd);
    }

    SPDLOG_LOGGER_DEBUG(logger, "Updated head. Old position: {}. New position: {}", oldHead,
                        m_head);

    return oldHead != m_head;
}

void SegmentTable::updateIdAllocator(void) {
    std::set<id::id_t> ids;
    for (std::size_t i = 0; i < m_segments.size(); i++) {
        const auto& segment = m_segments[i];
        if (segment.has_value()) {
            assert(segment.value().getId() == i &&
                   "Segment's id must match its index in m_segments");
            ids.emplace(static_cast<id::id_t>(i));
        }
    }
    m_idAllocator = id::IdAllocator();
    m_idAllocator.setUsedIds(ids);
}

void SegmentTable::throwOnInvalidId(const id_t segmentId) const {
    if (segmentId >= m_segments.size()) throw std::out_of_range("SegmentId out of range");
    if (!m_segments[segmentId].has_value()) throw std::invalid_argument("Segment uninitialized.");
}

}  // namespace memory

}  // namespace UnBARableAINS
