#ifndef PARTIAL_SEGMENT_TABLE_H_
#define PARTIAL_SEGMENT_TABLE_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <utility>

#include "memory/segment_table_entry.hpp"
#include "memory/segment_table_header.hpp"
#include "memory/shared_memory_types.h"

#include "serialization/segment_table_entry.tpp"

namespace UnBARableAINS {

namespace memory {

class PartialSegmentTable {
public:
    /// \brief Number of segments.
    inline static constexpr std::size_t c_segments_count = SegmentTableHeader::c_contiguous_count;

    using Entries = std::array<SegmentTableEntry, c_segments_count>;

    constexpr PartialSegmentTable(const position_t offset, const std::size_t index)
        : m_offset(offset)
        , m_header(c_segments_count)
        , m_index(index)
        , m_entries(buildEmptyEntries(index * c_segments_count,
                                      std::make_index_sequence<c_segments_count>{})) {}

    constexpr PartialSegmentTable(const position_t offset, const SegmentTableHeader header,
                                  Entries entries)
        : m_offset(offset)
        , m_header(header)
        , m_index(entries[0].getId() % c_segments_count)
        , m_entries(std::move(entries)) {
        if (entries[0].getId() % c_segments_count) {
            throw std::invalid_argument(
                std::format("First index {} is not a multiple of {} (c_segments_count).", m_index,
                            c_segments_count));
        }
        for (std::size_t i = 0; i < c_segments_count; i++) {
            const id_t expectedId = static_cast<id_t>(m_index * c_segments_count + i);
            const id_t actualId = m_entries[i].getId();
            if (actualId != expectedId) {
                throw std::invalid_argument(
                    std::format("Entry at position {} has invalid ID {}. Expected ID: {}", i,
                                actualId, expectedId));
            }
        }
    }

    constexpr PartialSegmentTable(const SegmentTableHeader header, Entries entries)
        : PartialSegmentTable(0, header, std::move(entries)) {}

    /**
     * \brief Check whether the table is full.
     * \returns Whether the table is full.
     */
    constexpr bool isFull(void) const {
        for (const auto& entry : m_entries) {
            if (entry.doesSegmentExist()) {
                return false;
            }
        }
        return true;
    }

    constexpr position_t getOffset(void) const { return m_offset; }

    constexpr const Entries& getEntries(void) const { return m_entries; }

    constexpr std::optional<position_t> nextPartialTablePosition(void) const {
        return static_cast<std::optional<position_t>>(m_header.getLink());
    }

private:
    template <std::size_t... Is>
        requires(sizeof...(Is) == c_segments_count)
    constexpr static Entries buildEmptyEntries(const std::size_t baseIndex,
                                               std::index_sequence<Is...>) {
        return {SegmentTableEntry(baseIndex + Is)...};
    }

    position_t m_offset;
    SegmentTableHeader m_header;
    std::size_t m_index;
    Entries m_entries;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // PARTIAL_SEGMENT_TABLE_H_
