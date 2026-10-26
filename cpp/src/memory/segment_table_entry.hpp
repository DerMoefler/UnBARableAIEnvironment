#ifndef SEGMENT_TABLE_ENTRY_H_
#define SEGMENT_TABLE_ENTRY_H_

#include <optional>

#include "memory/shared_memory_types.h"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Simple class to store an entry of the SegmentTable.
 *
 * The id of a segment is const and cannot be changed. The object itself also does not chang, the
 * only way of modification is creating a new object using the data of the old one (see \ref
 * updateLink).
 *
 * An entry simply looks like this:
 * \verbatim
 * [id][(optional) link]
 * \endverbatim
 */
class SegmentTableEntry {
public:
    /**
     * \brief Construct an entry.
     * \param id The id of the segment.
     * \param link Position of the segment in memory.
     */
    constexpr SegmentTableEntry(const id_t id, const std::optional<link_t> link = std::nullopt)
        : m_id(id)
        , m_link(link) {}

    /// \brief Defaulted comparison operator.
    constexpr bool operator==(const SegmentTableEntry& rhs) const = default;

    /**
     * \brief Construct a new entry with the same id but new link.
     * \param link New link.
     * \returns New entry object.
     */
    constexpr SegmentTableEntry updateLink(const std::optional<link_t> link) const {
        return SegmentTableEntry(m_id, link);
    }

    /// \brief Checks whether the segment has physical memory, i.e. m_link has a value.
    constexpr bool doesSegmentExist(void) const { return m_link.has_value(); }

    /// \brief Getter for id.
    constexpr id_t getId(void) const { return m_id; }

    /// \brief Getter for link.
    constexpr std::optional<link_t> getLink(void) const { return m_link; }

private:
    id_t m_id;
    std::optional<link_t> m_link;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SEGMENT_TABLE_ENTRY_H_
