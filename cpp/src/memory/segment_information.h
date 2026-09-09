#ifndef SEGMENT_INFORMATION_H_
#define SEGMENT_INFORMATION_H_

#include <cstddef>

#include "shared_memory_types.h"
#include "partially_linked_list_information.h"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Hold information about a Segment in shared memory.
 *
 * A instance of SegmentInformation can either be constructed as or set to be invalid. This
 * means that the Segment does not actually exist, meaning no information about it can be
 * queried. This is to allow reserving space for a SegmentInformation object without it holding
 * information about an actual Segment.
 * \see PartiallyLinkedListInformation
 * \todo Check whether inlined functions shouldnt have a validateState call.
 */
class SegmentInformation {
public:
    /**
     * \brief Constructs an invalid SegmentInformation instance.
     * \param id Id assigned to the invalid Segment.
     */
    SegmentInformation(const id_t id);

    /**
     * \brief Constructs a valid SegmentInformation instance with one partial segment.
     * \param id Id assigned to the Segment.
     * \param memoryStart Start of the segment in shared memory.
     * \param dataSize Data size for the first partial segment.
     */
    SegmentInformation(const id_t id, const position_t memoryStart, const size_t dataSize);

    /// \brief Default comparison operator.
    bool operator==(const SegmentInformation& other) const = default;

    /**
     * \brief Initialize a (previously) invalid segment.
     * \param memoryStart Start of the segment in shared memory.
     * \param dataSize Data size for the first partial segment.
     * \pre The instance must be invalid.
     * \post A valid instance with one partial segment.
     * \throws std::logic_error If called on a valid instance.
     */
    void initialize(const position_t memoryStart, const size_t dataSize);

    /**
     * \brief Check if the instance is valid.
     * \return Bool whether the instance holds valid information.
     */
    inline bool isValid(void) const { return m_valid; }

    /**
     * \brief Helper to determine whether a sequential read can be performed for the segment.
     * \param offset Relative offset within segment.
     * \param numBytes Number of bytes to read sequentially.
     *
     * This method essentially checks whether you can sequantially read \p numBytes while staying
     * inside the partial segment which \p offset lies in.
     * \note A sequential read of 0 bytes is considered unsafe.
     * \todo Check if 0 byte read being unsafe is actually what we want.
     */
    bool isSequentialReadSafe(position_t offset, size_t numBytes) const;

    /**
     * \brief Get the memory start of the List.
     * \return Memory start.
     */
    position_t getMemoryStart(void) const;

    /// \brief Get the data start of a partial segement.
    inline position_t getDataStart(size_t partialSegmentIndex) const {
        return m_information.getDataStart(partialSegmentIndex);
    }

    /// \brief Get the data size of a partial segment.
    inline size_t getDataSize(size_t partialSegmentIndex) const {
        return m_information.getDataSize(partialSegmentIndex);
    }

    /**
     * \brief Advance the Segment's head by the given amount.
     * \param increment Amount to advance the head by.
     */
    inline void advanceHead(size_t increment) { m_information.advanceHead(increment); }

    /**
     * \brief Set the head's position.
     * \param index Index into the data part of the segment to move the head to.
     */
    inline void setHead(position_t index) { m_information.setHead(index); }

    /**
     * \brief Get the validLength of the segment.
     * \returns ValidLength.
     * \see PartiallyLinkedListInformation::getOccupiedDataSize
     */
    inline size_t getValidLength(void) const { return m_information.getOccupiedDataSize(); }

    /**
     * \brief Get the position of the "validLength field" in the shm.
     * \returns Position of validLength for this segment.
     *
     * The validLength field specifies how much of this segment is actually readable data.
     */
    inline position_t getValidLengthPosition(void) const { return m_information.getHeaderStart(0); }

    /**
     * \brief Get the head's position.
     * \return The head's position_t.
     */
    inline position_t getHead(void) const { return m_information.getHead(); }

    /**
     * \brief Get the id.
     * \return The id_t.
     */
    inline id_t getId(void) const { return m_id; };

    /**
     * \brief Get the size.
     * \return Size of the list.
     * \see PartiallyLinkedListInformation::getSize
     */
    inline size_t getSize(void) const { return m_information.getSize(); }

    /**
     * \brief Get the data capacity.
     * \return Data capacity.
     * \see PartiallyLinkedListInformation::capacity
     */
    inline size_t getCapacity(void) const { return m_information.getCapacity(); }

    /// \brief Get the number of partial segments.
    inline size_t getPartialSegmentsCount(void) const {
        return m_information.getPartialSegmentsCount();
    }

    /**
     * \brief Check if the list is full.
     * \return Bool whether list is full.
     * \see PartiallyLinkedListInformation::isListFull
     */
    inline bool isListFull(void) const { return m_information.isListFull(); }

private:
    /**
     * \brief Validate that the object is valid.
     * \param state Bool whether the object has to be in (either valid or invalid).
     * \throws std::logic_error If the object is not in the correct state.
     */
    void validateState(bool state) const;

    id_t m_id;
    bool m_valid;
    PartiallyLinkedListInformation m_information;
};

}  // namespace memory

}  // namespace UnBARableAINS

#endif  // SEGMENT_INFORMATION_H_
