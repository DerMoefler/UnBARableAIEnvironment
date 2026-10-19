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
 * - total size of the segment, i.e. including the header.
 * - head position within the data part of the segment, i.e. how many bytes have actually been
 * written
 * - another segment id which this segment links to
 */
class SegmentHeader {
public:
    /// \brief Constant size of the header itself.
    inline static constexpr std::size_t c_size =
        sizeof(std::size_t) + sizeof(position_t) + sizeof(id_t);

    inline static constexpr position_t c_total_size_field_offset = 0;
    inline static constexpr position_t c_head_field_offset =
        c_total_size_field_offset + sizeof(std::size_t);
    inline static constexpr position_t c_next_segment_field_offset =
        c_head_field_offset + sizeof(position_t);

    /// \brief Alias for the view to the serialized data of a header.
    using HeaderView = std::span<const std::byte, c_size>;

    /**
     * \brief Constructs a header for a segment that does not yet actually exist.
     */
    constexpr SegmentHeader(void)
        : m_totalSize(0)
        , m_dataSize(0)
        , m_head(0)
        , m_nextSegmentId(std::nullopt) {}

    /**
     * \brief Constructs a header for a segment that has not yet been written to.
     * \param dataSize Size of the segment's data part.
     */
    constexpr SegmentHeader(const std::size_t dataSize)
        : m_totalSize(dataSize + c_size)
        , m_dataSize(dataSize)
        , m_head(0)
        , m_nextSegmentId(std::nullopt) {}

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

    /// \brief Getter for total size.
    constexpr std::size_t getTotalSize(void) const { return m_totalSize; }

    /// \brief Getter for head.
    constexpr position_t getHead(void) const { return m_head; }

private:
    std::size_t m_totalSize;
    std::size_t m_dataSize;
    position_t m_head;

    std::optional<id_t> m_nextSegmentId;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SEGMENT_HEADER_H_
