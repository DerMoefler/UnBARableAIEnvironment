#ifndef SEGMENT_INFORMATION_H_
#define SEGMENT_INFORMATION_H_

#include <cassert>
#include <cstddef>

#include "memory/shared_memory_types.h"
#include "memory/segment_header.hpp"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Stores information about a segment in shared memory.
 *
 * Consists of a header indicating the total size occupied in memory (including the entire header),
 * then the valid data length, i.e. the head position within the segment's data section, and finally
 * an optional link to another segment id, allowing segments to be linked together.
 *
 * \note The full head position actually refers to the first byte directly AFTER the segment,
 * meaning the actual range for the segment is actually [m_memoryStart, fullHeadPosition).
 */
class SegmentInformation {
public:
    /// \brief Constant size of the header.
    inline static constexpr std::size_t c_header_size = SegmentHeader::c_size;

    using HeaderView = SegmentHeader::HeaderView;

    /**
     * \brief Constructor for an invalid segment, i.e. one that is not in use.
     * \param id Id assigned to the segment.
     *
     * Zero initializes all sizes and even memory start.
     */
    constexpr SegmentInformation(const id_t id)
        : m_id(id)
        , m_isValid(false)
        , m_memoryStart(0)
        , m_header() {}

    /**
     * \brief Constructor for a valid segment in memory, i.e. one that actually stores data.
     * \param id Id assigned to the segment.
     * \param memoryStart Start of the header in the shared memory.
     * \param dataSize Payload size of the segment. (Can be 0).
     */
    constexpr SegmentInformation(const id_t id, const position_t memoryStart,
                                 const std::size_t dataSize)
        : m_id(id)
        , m_isValid(true)
        , m_memoryStart(memoryStart)
        , m_header(SegmentHeader(dataSize)) {}

    /**
     * \brief Create a valid segment from an invalid one.
     * \param memoryStart Start of the header in the shared memory.
     * \param dataSize Paylod size of the segment.
     * \returns Valid object constructed using the same id.
     */
    [[nodiscard]]
    constexpr SegmentInformation initialize(const position_t memoryStart,
                                            const std::size_t dataSize) const {
        assert(!m_isValid);
        return SegmentInformation(m_id, memoryStart, dataSize);
    }

    /**
     * \brief Quickly invalidate a segment while keeping its layout information intact and
     * queryable.
     * \returns Invalid object with the same data.
     */
    [[nodiscard]]
    constexpr SegmentInformation invalidate(void) const {
        assert(m_isValid);
        SegmentInformation result = *this;
        result.m_isValid = false;
        return result;
    }

    /// \brief Default comparison.
    constexpr bool operator==(const SegmentInformation& other) const = default;

    /**
     * brief Get the position of the first data byte in the shared memory.
     * \returns The data position.
     */
    constexpr position_t dataStart(void) const {
        return m_memoryStart + static_cast<position_t>(c_header_size);
    }

    /**
     * \brief Get the position of the head in the shared memory.
     * \returns The head position.
     */
    constexpr position_t headPosition(void) const {
        return m_memoryStart + static_cast<position_t>(c_header_size) + m_header.getHead();
    }

    /**
     * \brief Get the position of the head if the segment was full, i.e. the end of the segment in
     * memory.
     * \returns Position of the head if the segment was full.
     */
    constexpr position_t fullHeadPosition(void) const {
        return m_memoryStart + static_cast<position_t>(m_header.getTotalSize());
    }

    /**
     * \brief Get the number of bytes already written to the segment.
     * \return Number of bytes.
     */
    constexpr std::size_t validLength(void) const {
        return static_cast<std::size_t>(m_header.getHead());
    }

    /**
     * \brief Advance the head by \p increment.
     * \param increment Number of bytes to increment the head by.
     * \throws std::logic_error if the increment is larger than \ref SegmentHeader::remainingSize().
     * \see SegmentHeader::advanceHead()
     */
    constexpr void advanceHead(const std::size_t increment) { m_header.advanceHead(increment); }

    /// \brief Get total size the segment occupies in memory, i.e. including its header.
    constexpr std::size_t totalSize(void) const { return m_header.getTotalSize(); }

    /**
     * \brief Checks if the segment is valid.
     * \returns Whether the segment is valid.
     */
    constexpr bool isValid(void) const { return m_isValid; }

    /// \brief Getter for id.
    constexpr id_t getId(void) const { return m_id; }

    /// \brief Getter for memory start.
    constexpr position_t getMemoryStart(void) const { return m_memoryStart; }

private:
    id_t m_id;
    bool m_isValid;
    position_t m_memoryStart;
    SegmentHeader m_header;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SEGMENT_INFORMATION_H_
