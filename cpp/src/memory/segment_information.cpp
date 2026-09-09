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
