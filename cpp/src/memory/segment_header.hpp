#ifndef SEGMENT_HEADER_H_
#define SEGMENT_HEADER_H_

#include <cstddef>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>

#include "memory/shared_memory_types.h"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Representation for the deserialized segment header.
 *
 * A segment stores information about itself in the header. These information are
 * - data size of the segment, i.e. excluding the header.
 * - head position within the data part of the segment, i.e. how many bytes have actually been
 * written
 * - another segment id which this segment links to
 *
 * To get the size of the header, use the SerializeInformation specialization.
 */
class SegmentHeader {
public:
    /**
     * \brief Constructs a header for a segment that does not yet actually exist.
     */
    constexpr SegmentHeader(void)
        : m_dataSize(0)
        , m_head(0)
        , m_nextSegmentId(std::nullopt) {}

    /**
     * \brief Constructs a header for a segment that has not yet been written to.
     * \param dataSize Size of the segment's data part.
     */
    constexpr SegmentHeader(const std::size_t dataSize)
        : m_dataSize(dataSize)
        , m_head(0)
        , m_nextSegmentId(std::nullopt) {}

    constexpr SegmentHeader(const std::size_t dataSize, const memory::position_t head,
                            const std::optional<memory::id_t> nextSegmentId)
        : m_dataSize(dataSize)
        , m_head(head)
        , m_nextSegmentId(nextSegmentId) {
        if (m_head > m_dataSize) {
            throw std::logic_error(
                std::format("Head cannot be at position {} because the maximum data size is {}.",
                            m_head, m_dataSize));
        }
    }

    /// \brief Default comparison.
    constexpr bool operator==(const SegmentHeader& other) const = default;

    /**
     * \brief Advance the head by \p increment.
     * \param increment Number of bytes to increment the head by.
     * \throws std::logic_error if the increment is larger than \ref remainingSize().
     */
    constexpr void advanceHead(const std::size_t increment) {
        if (increment > remainingSize()) {
            throw std::logic_error(std::format(
                "Cannot advance head by {} bytes because the remaining size is only {} bytes",
                increment, remainingSize()));
        }
        m_head += increment;
    }

    /**
     * \brief Calculate the remaining number of bytes that can be written to the segment.
     * \returns Number of bytes.
     */
    constexpr std::size_t remainingSize(void) const {
        return m_dataSize - static_cast<std::size_t>(m_head);
    }

    /// \brief Getter for dataSize.
    constexpr std::size_t getDataSize(void) const { return m_dataSize; }

    /// \brief Getter for head.
    constexpr position_t getHead(void) const { return m_head; }

    constexpr std::optional<id_t> getNextSegmentId(void) const { return m_nextSegmentId; }

private:
    std::size_t m_dataSize;
    position_t m_head;

    std::optional<id_t> m_nextSegmentId;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SEGMENT_HEADER_H_
