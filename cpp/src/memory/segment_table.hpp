#ifndef SEGMENT_TABLE_H_
#define SEGMENT_TABLE_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>
#include <utility>

#include "memory/shared_memory_types.h"
#include "memory/segment_information.hpp"

namespace UnBARableAINS {

namespace memory {

namespace detail {

/**
 * \brief Handles a contiguous part of a \ref SegmentTable.
 */
class PartialSegmentTable {
public:
    /// \brief Number of segments.
    inline static constexpr std::size_t c_segments_count = 32;

    /// \brief Size of an entry consisting of the id and a link.
    inline static constexpr std::size_t c_entry_size = sizeof(id_t) + sizeof(link_t);

    /// \brief Size of a partial segment table.
    inline static constexpr std::size_t c_size = (c_segments_count + 1) * c_entry_size;

    using SerializedView = std::span<const std::byte, c_size>;

    using Segments = std::array<SegmentInformation, c_segments_count>;

    /**
     * \brief Construct a PartialSegmentTable.
     * \param offset Offset in SharedMemory where the PartialSegmentTable begins.
     * \param index Index of the PartialSegmentTable in \ref SegmentTable.
     *
     * The \ref c_segment_count dictates the number of Segments per PartialSegmentTable. The \p
     * index specifies the index of the entire PartialSegmentTable in the SegmentTable. This means,
     * index = 0 creates SegmentInformation for ids [0, 31], index=1 for ids [32, 63] etc.
     */
    constexpr PartialSegmentTable(position_t offset, std::size_t index)
        : m_offset(offset)
        , m_segments(buildSegments(std::make_index_sequence<c_segments_count>{})) {}

    PartialSegmentTable(position_t offset, SerializedView serialized);

    bool updateLinks(SerializedView serialized);

    constexpr std::size_t validSegmentCount(void) const {
        std::size_t count{0};
        for (const auto& segment : m_segments) {
            if (segment.isValid()) {
                count++;
            }
        }
        return count;
    }

    position_t getLinkPosition(void) const;

private:
    template <std::size_t... Is>
        requires(sizeof...(Is) == c_segments_count)
    constexpr static std::array<SegmentInformation, c_segments_count> buildSegments(
        std::index_sequence<Is...>) {
        return {SegmentInformation(Is)...};
    }

    inline static id_t getIdAtIndex(SerializedView serialized, std::size_t index) {
        assert(index < c_segments_count);
        return deserializeAtPosition(serialized, static_cast<position_t>(index) * c_entry_size);
    }

    inline static link_t getLinkAtIndex(SerializedView serialized, std::size_t index) {
        assert(index < c_segments_count);
        return static_cast<link_t>(deserializeAtPosition(
            serialized, static_cast<position_t>(index) * c_entry_size + sizeof(id_t)));
    }

    static id_t deserializeAtPosition(SerializedView, position_t position);

    /// \brief Offset in memory.
    position_t m_offset;
    /// \brief Stores Segments (Information) where index corresponds to Segment's id.
    Segments m_segments;
};

}  // namespace detail

class SegmentTable {
private:
    using PartialSegmentTable = detail::PartialSegmentTable;

public:
    /// \brief Number of contiguous segments.
    inline static constexpr std::size_t c_contiguous_segments_count =
        PartialSegmentTable::c_segments_count;

    using SerializePartialSegmentTableView = PartialSegmentTable::SerializedView;

    /// \brief The id used to signal that the next entry is not an offset for a data segment but
    /// rather the offset to the next partial segment table.
    inline static constexpr id_t c_partial_table_link_id = 0xFE'FC'BA'98;  //'76'54'32'10

    inline SegmentTable(void)
        : m_partialTables() {}

    inline SegmentTable(position_t offset)
        : m_partialTables{PartialSegmentTable(offset, 0)} {}

    id_t allocate(void);
    void allocate(const id_t requestedId);

    void release(const id_t segmentId);

    void extend(void);

    std::size_t validSegmentCount(void) const;

private:
    std::vector<PartialSegmentTable> m_partialTables;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SEGMENT_TABLE_H_
