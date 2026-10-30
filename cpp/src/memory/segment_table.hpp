#ifndef SEGMENT_TABLE_H_
#define SEGMENT_TABLE_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>
#include <utility>

#include "id/id_allocator.hpp"
#include "memory/partial_segment_table.hpp"
#include "memory/shared_memory_types.h"
#include "memory/shm_header.hpp"
#include "memory/segment_information.hpp"
#include "serialization/trivial_serializable.hpp"

#include "serialization/partial_segment_table.tpp"
#include "serialization/shm_header.tpp"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Handles the allocation and deallocation of new segments.
 *
 * This class is designed to hold the information about segments. The metadata (i.e. id and
 * position) reside in one or multiple PartialSegmentTables. Instances of \ref SegmentInformation
 * then store information about the actual segments, like their datasize etc. This class does not
 * perform physical modifications, i.e. allocating a Segment here does require the SharedMemoryImpl
 * class to then actually write the SegmentHeader. This class is information only.
 */
class SegmentTable {
public:
    /// \brief Construct an empty SegmentTable (no PartialSegmentTable, no SegmentInformation).
    SegmentTable(void) = default;

    // TODO implement
    SegmentTable(std::vector<PartialSegmentTable> tables,
                 std::vector<std::optional<SegmentHeader>> headers);

    /// \brief Default comparison operator
    bool operator==(const SegmentTable& rhs) const = default;

    /**
     * \brief Allocates a new segment of the size \p segmentDataSize.
     * \param segmentDataSize Number of data bytes for the new Segment.
     * \returns Corresponding instance of SegmentInformation.
     */
    [[nodiscard("The new SegmentInformation's header must be written to the physical shm.")]]
    const SegmentInformation& allocate(const std::size_t segmentDataSize);

    /**
     * \brief Releases the segment with the id \p segmentId.
     * \param segmentId Id of the segment.
     * \throws See \ref getSegmentInformation.
     */
    void release(const id_t segmentId);

    /**
     * \brief Check whether the table is full.
     * \returns Whether the table is full.
     */
    bool isFull(void) const;

    [[nodiscard("The new PartialSegmentTable must be written to the physical shm.")]]
    const PartialSegmentTable& extend(void);

    /**
     * \brief Advance the head of the segment with id \p segmentId.
     * \param segmentId Id of the segment
     * \param increment Number of bytes to move the head by.
     * \throws See \ref getSegmentInformation.
     */
    void advanceSegmentInformationHead(const id_t segmentId, const std::size_t increment);

    /**
     * \brief Get the number of valid segments.
     * \returns Number of valid segemnts.
     */
    [[nodiscard]]
    std::size_t validSegmentsCount(void) const;

    /**
     * \brief Get the SegmentInformation for \p segmentId.
     * \param segmentId Id of the requested segment.
     * \returns SegmentInformation.
     * \throws std::invalid_argument if m_segments[segmentId] == std::nullopt.
     * \throws std::out_of_range if segmentId > m_segments.size()
     */
    [[nodiscard]]
    const SegmentInformation& getSegmentInformation(const id_t segmentId) const;

    /**
     * \brief Get the position in memory of the SegmentEntry for the \p segmentId.
     * \param segmentId Id of the segment.
     * \throws std::out_of_range if segmentId > m_segments.size().
     * \returns Position.
     */
    [[nodiscard]]
    const position_t getSegmentEntryPosition(const id_t segmentId) const;

    [[nodiscard]]
    position_t getHead(void) const {
        return m_head;
    }

    [[nodiscard]]
    const std::vector<PartialSegmentTable>& getPartialSegmentTables(void) const {
        return m_partialSegmentTables;
    }

private:
    static constexpr std::size_t c_partial_table_size =
        serialization::getSerializableSize(std::type_identity<PartialSegmentTable>{});

    /**
     * \brief Helper to allocate a chunk of memory.
     * \param chunkSize Number of bytes for the chunk.
     * \returns position_t of the first byte.
     */
    [[nodiscard]]
    position_t allocateChunk(const std::size_t chunkSize);

    /**
     * \brief Updates the head position, based off partial tables and segments.
     * \returns Whether the head changed.
     */
    bool updateHead(void);

    void updateIdAllocator(void);

    /**
     * \brief Helper to throw if a segmentId is invalid.
     * \param segmentId Id to check.
     * \throws std::invalid_argument if m_segments[segmentId] == std::nullopt.
     * \throws std::out_of_range if segmentId > m_segments.size()
     */
    void throwOnInvalidId(const id_t segmentId) const;

    static constexpr std::size_t getTableIndex(const id_t id) {
        return id / PartialSegmentTable::c_segments_count;
    }

    static constexpr std::size_t getIntableIndex(const id_t id) {
        return id % PartialSegmentTable::c_segments_count;
    }

    /// \brief Head of the shared memory. Starts after the size of \ref ShmHeader.
    position_t m_head = static_cast<position_t>(
        serialization::getSerializableSize(std::type_identity<ShmHeader>{}));

    std::vector<PartialSegmentTable> m_partialSegmentTables = {};
    std::vector<std::optional<SegmentInformation>> m_segments = {};
    id::IdAllocator m_idAllocator{};
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SEGMENT_TABLE_H_
