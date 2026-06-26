#pragma once
#include <cstddef>
#include <vector>
#include <stdexcept>

#include "shared_memory_types.h"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Stores positional information about a partially linked list in shared memory.
 */
class PartiallyLinkedListInformation {
public:
    /**
     * \brief Error class used when an index calculated by \ref getIndexPosition would be in the header / trailer.
     */
    class IndexInSegmentInfo : public std::invalid_argument {
    public:
        /**
         * \brief Simple constructor.
         * \param partialSegmentIndex Index to the relevant partial segment.
         * \param offset Offset relative to the start (i.e. header) of the partial segment.
         * \param headerSize Header size of the partial segment.
         * \param partialSegmentSize Size of the partial segment.
         */
        IndexInSegmentInfo(
            std::size_t partialSegmentIndex,
            std::size_t offset,
            std::size_t headerSize
        )
            : std::invalid_argument(
                "Position lies within header / link section of partial segment."
            ),
            m_partialSegmentIndex(partialSegmentIndex),
            m_offset(offset),
            m_headerSize(headerSize)
        {}
        /// \brief Getter for partial segment index.
        inline std::size_t getPartialSegmentIndex() const { return m_partialSegmentIndex; }
        /// \brief Getter for data offset.
        inline std::size_t getOffset() const { return m_offset; }
        /// \brief Getter for header size.
        inline std::size_t getHeaderSize() const { return m_headerSize; }

    private:
        std::size_t m_partialSegmentIndex;
        std::size_t m_offset;
        std::size_t m_headerSize;
    };

    /// \brief Length of the link at the end of each partial segment so link to the next.
    inline static constexpr size_t c_link_size = sizeof(position_t);
    /**
     * \brief Construct a partially linked list, starting with one partial segment at the memory start.
     * \param memoryStart Start of the list which is therefore also the start of the first partial segment.
     * \param headerSize Size of the header for the first partial segement.
     * \param dataSize Size of the data for the first partial segement.
     */
    PartiallyLinkedListInformation(const position_t memoryStart, const size_t headerSize, const size_t dataSize);
    
    /**
     * \brief Add a new partial segment.
     * \param offset Offset in shared memory for the new partial segment.
     * \param headerSize Size of the header for the new partial segement.
     * \param dataSize Size of the data for the new partial segement.
     */
    void extend(const position_t offset, const size_t headerSize, const size_t dataSize);

    /**
     * \brief Find the position_t (relative to the shared memory's start) from the position of the data inside the list.
     * \param index Data position inside the entire list.
     * \return The position in shared memory.
     * \throws std::out_of_range If the index is larger than the data size.
     * \see Inverse: \ref getIndexPosition.
     */
    position_t getDataPosition(const position_t index) const;

    /**
     * \brief Find the index (into the data portion of the Segment) from a position into shared memory.
     * \param position Position in the shared memory.
     * \return The index into the Segment's data portion.
     * \throws std::out_of_range If the position is not within a partial segment.
     * \throws std::invalid_argument If the position is within the header or trailer of a partial segment.
     * \see Inverse: \ref getDataPosition.
     */
    position_t getIndexPosition(const position_t position) const;

    /**
     * \brief Find the position_t (relative to the shared memory's start) of the header of some partial segment.
     * \param partialSegmentIndex The partial segment's index.
     */
    position_t getHeaderStart(const size_t partialSegmentIndex) const;
    
    /**
     * \brief Find the position_t (relative to the shared memory's start) of the data of some partial segment.
     * \param partialSegmentIndex The partial segment's index.
     */
    position_t getDataStart(const size_t partialSegmentIndex) const;

    /**
     * \brief Advance the list's head by the given amount.
     * \param increment Amount to advance the head by.
     */
    void advanceHead(size_t increment);

    /**
     * \brief Set the head's position.
     * \param index Index into the data part of the list to move the head to.
     */
    void setHead(position_t index);

    /**
     * \brief Get the head's position.
     * \return The head's position_t.
     */
    inline position_t getHead(void) const { return m_head; };

    /**
     * \brief Find the size of a partial segment.
     * \param partialSegmentIndex The partial segment's index.
     */
    size_t getHeaderSize(const size_t partialSegmentIndex) const;

    /**
     * \brief Get the entire size the list occupies in memory.
     * \return Size.
     */
    inline size_t getSize(void) const { return m_size; };
    
    /**
     * \brief Get the memory start of the List.
     * \return Memory start.
     */
    inline position_t getMemoryStart(void) const { return m_memoryStart; };

    /**
     * \brief Get the data capacity of the list.
     * \return Data capacity.
     */
    size_t getCapacity(void) const;
    
    /**
     * \brief Checks whether the list is full or not.
     * \return Bool if the head is at \ref getFullHeadPosition.
     */
    inline bool isListFull(void) const { return m_head == getFullHeadPosition(); }
private:
    /**
     * \brief Simple validation that the partial segment with that index exists.
     * \throws std::out_of_range If the index does not exist.
     */ 
    void validatePartialSegmentIndex(const size_t partialSegmentIndex) const;

    /**
     * \brief Calculates the position of the head when the list is full.
     * \return Position: Last partial element's link (value of the link must be 0x00).
     */
    position_t getFullHeadPosition(void) const;
    
    /// \brief The position of the head
    position_t                  m_head;
    /// \brief 
    position_t                  m_memoryStart;
    /// \brief The total size occupied in memory
    size_t                      m_size          = 0;
    /// \brief Offsets of the partial elements.
    std::vector<position_t>     m_offsets       = {};
    /// \brief Sizes of the partial elements.
    std::vector<size_t>         m_sizes         = {};
    /// \brief Data sizes (capacity) of the partial elements.
    std::vector<size_t>         m_dataSizes     = {};
};

}; // namespace memory

}; // namespace UnBARableAI
