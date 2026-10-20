#ifndef SEGMENT_TABLE_HEADER_H_
#define SEGMENT_TABLE_HEADER_H_

#include <optional>

#include "memory/shared_memory_types.h"

namespace UnBARableAINS {

namespace memory {

class SegmentTableHeader {
public:
    constexpr SegmentTableHeader() {}

private:
    std::size_t m_totalSize;
    std::size_t m_numElements;

    std::optional<link_t> m_link;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SEGMENT_TABLE_HEADER_H_
