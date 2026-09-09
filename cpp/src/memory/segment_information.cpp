#include "segment_information.h"

namespace UnBARableAINS {

namespace memory {

SegmentInformation::SegmentInformation(const id_t id)
    : m_id(id)
    , m_valid(false)
    , m_information(0, 0, 0) {}

SegmentInformation::SegmentInformation(const id_t id, const position_t memoryStart,
                                       const size_t dataSize)
    : SegmentInformation(id) {
    initialize(memoryStart, dataSize);
}

void SegmentInformation::initialize(const position_t memoryStart, const size_t dataSize) {
    validateState(false);
    m_information = PartiallyLinkedListInformation{memoryStart, 2 * sizeof(size_t), dataSize};
    m_valid = true;
}

bool SegmentInformation::isSequentialReadSafe(position_t position, size_t numBytes) const {
    if (!numBytes) {
        return false;
    }
    auto startIndexOpt = m_information.getPartialSegmentIndex(position);

    // We want to read numBytes, but the end position to check is actually one before position +
    // numBytes. Suppose we were at the lastByte of a partial segment and wanted to read it but the
    // segment continued further, i.e. there comes a partial segment after the current one.
    // Now, lastByte + 1 would actually mean the firstByte of the nextPartialSegment. That is not
    // what we are asking though. We want to know if we can read that one byte sequentially, which
    // (in this case trivially) we absolutely can. This is why when choosing the end position, we
    // must actually subtract one.
    position_t endPosition = position + numBytes - 1;
    auto endIndexOpt = m_information.getPartialSegmentIndex(endPosition);
    if (startIndexOpt.has_value() && endIndexOpt.has_value()) {
        return startIndexOpt.value() == endIndexOpt.value();
    }
    else {
        return false;
    }
}

position_t SegmentInformation::getMemoryStart(void) const {
    validateState(true);
    return m_information.getMemoryStart();
}

void SegmentInformation::validateState(bool state) const {
    if (m_valid != state) {
        std::string_view message;
        if (m_valid) {
            message = "Object has to be invalid for the request operation, but is valid.";
        }
        else {
            message = "Object has to be valid for the request operation, but is invalid.";
        }
        throw std::logic_error(message.data());
    }
}

}  // namespace memory

}  // namespace UnBARableAINS
