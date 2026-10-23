#ifndef SEGMENT_TABLE_HEADER_H_
#define SEGMENT_TABLE_HEADER_H_

#include <optional>
#include <format>
#include <stdexcept>

#include "memory/shared_memory_types.h"
#include "memory/segment_table_entry.hpp"
#include "serialization/trivial_serializable.hpp"

#include "serialization/segment_table_entry.tpp"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Immutable header for a SegmentTable.
 *
 * This class essentially only contains a link to the beginning of the next header, allowing to link
 * PartialSegmentTable instances. Every single header must have a count of \c_contiguous_count.
 */
class SegmentTableHeader {
public:
    /// \brief Number of contiguous segments.
    inline static constexpr std::size_t c_contiguous_count = 32;

    /// \brief Size of a SegmentTableEntry.
    static constexpr std::size_t c_entry_size =
        serialization::getSerializableSize(std::type_identity<SegmentTableEntry>{});

    /**
     * \brief Default constructor.
     * \param numElements Number of elements, must be equal to c_contiguous_count.
     * \param link position_t where the table continues further.
     *
     * \throws std::invalid_argument if numElements != c_contiguous_count
     *
     * This constructor is essentially just used to set the link, however numElements is provided to
     * allow easily deserializing and throwing for an invalid header.
     */
    constexpr SegmentTableHeader(const std::size_t numElements,
                                 const std::optional<link_t> link = std::nullopt)
        : m_link(link) {
        if (numElements != c_contiguous_count) {
            throw std::invalid_argument(
                std::format("Invalid header. The count is a compile time constant, expected to be "
                            "{}. Actual encountered value is {}",
                            c_contiguous_count, numElements));
        }
    }

    /**
     * \brief Create a new object with a different link.
     * \param link Link for the new object.
     * \returns A new object with that link.
     */
    [[nodiscard(
        "Object is immutable. Instead, a new object for the requested operation is returned.")]]
    constexpr SegmentTableHeader updateLink(const std::optional<link_t> link) {
        return SegmentTableHeader(c_contiguous_count, link);
    }

    /// \brief Getter for link.
    constexpr std::optional<link_t> getLink(void) const { return m_link; }

private:
    const std::optional<link_t> m_link;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SEGMENT_TABLE_HEADER_H_
