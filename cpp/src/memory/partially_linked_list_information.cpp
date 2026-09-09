#include "partially_linked_list_information.h"

#include <stdexcept>

namespace UnBARableAINS {

namespace memory {

PartiallyLinkedListInformation::PartiallyLinkedListInformation(const position_t memoryStart,
                                                               const size_t headerSize,
                                                               const size_t dataSize)
    : m_memoryStart(memoryStart) {
    extend(memoryStart, headerSize, dataSize);
    m_head = memoryStart + headerSize;
}

void PartiallyLinkedListInformation::extend(const position_t offset, const size_t headerSize,
                                            const size_t dataSize) {
    // Move head before extending the list if necessary
    if (m_offsets.size() && isListFull()) {
        m_head = offset + headerSize;
    }
    const size_t totalSize = headerSize + dataSize + c_link_size;
    m_offsets.push_back(offset);
    m_dataSizes.push_back(dataSize);
    m_sizes.push_back(totalSize);
    m_size += totalSize;
}

position_t PartiallyLinkedListInformation::getDataPosition(const position_t index) const {
    if (index >= getCapacity()) {
        throw std::out_of_range("Index exceeds capacity");
    }
    size_t partialSegmentIndex = 0;
    size_t previousDataSize = 0;
    for (int i = 0; i < m_dataSizes.size(); i++) {
        if (previousDataSize + m_dataSizes[i] > index) {
            break;
        }
        partialSegmentIndex++;
        previousDataSize += m_dataSizes[i];
    }
    const size_t indexInSegment = index - previousDataSize;
    return getDataStart(partialSegmentIndex) + indexInSegment;
}

position_t PartiallyLinkedListInformation::getIndexPosition(const position_t position) const {
    // Get the first partial segment offset larger than position
    auto it = std::lower_bound(m_offsets.begin(), m_offsets.end(), position);
    if (it == m_offsets.begin()) {
        throw std::out_of_range("Position not within any partial segments.");
    }
    --it;
    const size_t partialSegmentIndex = std::distance(m_offsets.begin(), it);
    const size_t offset = position - *it;
    if (offset >= m_sizes[partialSegmentIndex]) {
        throw std::out_of_range("Position not within any partial segments.");
    }
    const size_t headerSize = getHeaderSize(partialSegmentIndex);
    if (offset < headerSize || offset >= (headerSize + m_dataSizes[partialSegmentIndex])) {
        throw IndexInSegmentInfo(partialSegmentIndex, offset, headerSize);
    }
    position_t dataIndex = 0;
    for (int i = 0; i < partialSegmentIndex; i++) {
        dataIndex += m_dataSizes[i];
    }
    return dataIndex + offset - headerSize;
}

position_t PartiallyLinkedListInformation::getHeaderStart(const size_t partialSegmentIndex) const {
    validatePartialSegmentIndex(partialSegmentIndex);
    return m_offsets[partialSegmentIndex];
}

position_t PartiallyLinkedListInformation::getDataStart(const size_t partialSegmentIndex) const {
    validatePartialSegmentIndex(partialSegmentIndex);
    return m_offsets[partialSegmentIndex] + getHeaderSize(partialSegmentIndex);
}

std::optional<size_t> PartiallyLinkedListInformation::getPartialSegmentIndex(
    position_t offset) const {
    if (offset > getCapacity()) {
        return std::nullopt;
    }
    // TODO refactor this method and getDataPosition to use the same basis (since its the same code
    // for the majority).
    size_t partialSegmentIndex = 0;
    size_t previousDataSize = 0;
    for (size_t i = 0; i < m_dataSizes.size(); i++) {
        if (previousDataSize + m_dataSizes[i] > offset) {
            break;
        }
        partialSegmentIndex++;
        previousDataSize += m_dataSizes[i];
    }
    return partialSegmentIndex;
}

size_t PartiallyLinkedListInformation::getOccupiedDataSize(void) const {
    if (isListFull()) {
        return getCapacity();
    }
    else {
        return getIndexPosition(m_head);
    }
}

void PartiallyLinkedListInformation::advanceHead(size_t increment) {
    position_t index = getIndexPosition(m_head);
    index += increment;
    try {
        m_head = getDataPosition(index);
    } catch (const std::out_of_range& e) {
        if (getFullHeadPosition() - 1 == getDataPosition(index - 1)) {
            m_head = getFullHeadPosition();
        }
    }
}

size_t PartiallyLinkedListInformation::getHeaderSize(const size_t partialSegmentIndex) const {
    validatePartialSegmentIndex(partialSegmentIndex);
    return m_sizes[partialSegmentIndex] - m_dataSizes[partialSegmentIndex] - c_link_size;
}

size_t PartiallyLinkedListInformation::getDataSize(size_t partialSegmentIndex) const {
    validatePartialSegmentIndex(partialSegmentIndex);
    return m_dataSizes[partialSegmentIndex];
}

size_t PartiallyLinkedListInformation::getCapacity(void) const {
    size_t capacity = 0;
    for (const auto& size : m_dataSizes) capacity += size;
    return capacity;
}

void PartiallyLinkedListInformation::validatePartialSegmentIndex(
    const size_t partialSegmentIndex) const {
    if (partialSegmentIndex >= m_offsets.size()) {
        throw std::out_of_range("PartialSegmentIndex is out of range");
    }
}

position_t PartiallyLinkedListInformation::getFullHeadPosition(void) const {
    size_t lastPartialSegmentIndex = m_offsets.size() - 1;
    return m_offsets[lastPartialSegmentIndex] + getHeaderSize(lastPartialSegmentIndex) +
           m_dataSizes[lastPartialSegmentIndex];
}

};  // namespace memory

};  // namespace UnBARableAINS
