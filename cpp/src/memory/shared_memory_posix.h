#pragma once
#include <cstddef>
#include <string>
#include <cstdint>
#include <vector>
#include <span>
#include <cstring>
#include <concepts>
#include <endian.h>

#include "../memory/shared_memory_types.h"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Implements a \ref SharedMemoryImpl using POSIX shared memory.
 * \todo fix use of uint64_t and id_t to refer to an index / size type.
 * 
 * \todo document
 * \image html shared_memory_layout.svg
 * 
 * People seem to prefer POSIX shared memory over system V's, which is used by \ref SharedMemoryUnix.
 */
class SharedMemoryPosix {
    static_assert(sizeof(size_t) == 8, "Invalid bytelength for type 'size_t'");
public:
    /// \brief Alias for a view to some data (e.g. to write to a segment).
    using DataView = const std::span<const std::byte>;

    /**
     * \brief Stores positional information about a partially linked list in shared memory.
     */
    class PartiallyLinkedListInformation {
    public:
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
         */
        size_t getCapacity(void) const;
        
    private:
        /**
         * \brief Simple validation that the partial segment with that index exists.
         * \throws std::out_of_range If the index does not exist.
         */ 
        void validatePartialSegmentIndex(const size_t partialSegmentIndex) const;
        
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

    /**
     * \brief Hold information about a Segment in shared memory.
     * 
     * A instance of SegmentInformation can either be constructed as or set to be invalid. This means that the Segment does not actually exist,
     * meaning no information about it can be queried. This is to allow reserving space for a SegmentInformation object without it holding
     * information about an actual Segment.
     * \see PartiallyLinkedListInformation
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

        /**
         * \brief Initialize an invalid segment.
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
         * \brief Get the memory start of the List.
         * \return Memory start.
         */
        position_t getMemoryStart(void) const;

        /**
         * \brief Advance the Segment's head by the given amount.
         * \param increment Amount to advance the head by.
         */
        inline void advanceHead(size_t increment) { m_information.advanceHead(increment); };

        /**
         * \brief Set the head's position.
         * \param index Index into the data part of the segment to move the head to.
         */
        inline void setHead(position_t index) { m_information.setHead(index); };

        /**
         * \brief Get the head's position.
         * \return The head's position_t.
         */
        inline position_t getHead(void) const { return m_information.getHead(); };

        /**
         * \brief Get the id.
         * \return The id_t.
         */
        inline id_t getId(void) const { return m_id; };

        /**
         * \brief Get the size.
         * \return The id_t.
         * \see PartiallyLinkedListInformation::getSize
         */
        inline size_t getSize(void) const { return m_information.getSize(); };

    private:
        /**
         * \brief Validate that the object is valid.
         * \param state Bool whether the object has to be in (either valid or invalid).
         * \throws std::logic_error If the object is not in the correct state.
         */
        void validateState(bool state) const;
        
        id_t                                    m_id;
        bool                                    m_valid;
        PartiallyLinkedListInformation          m_information;
    };

    /// \brief The id used to signal that the next entry is not an offset for a data segment but rather the offset to the next partial segment table.
    inline static constexpr id_t   c_partial_table_link_id = 0xFE'DC'BA'98;//'76'54'32'10;

    /**
    * \brief Number of contiguous segments in the partially linked list.
    * The last element is a relative pointer to next array. \todo image
    */ 
    inline static constexpr uint32_t    c_contiguous_segment_count = 2;

    /// \brief The id used to signal that the following bytes compose a \ref position_t to where the segment continues.
    inline static constexpr id_t   c_segment_link_id       = 0xAA'AA'AA'AA;

    /**
     * \brief Acquire shared memory using shm_open().
     * \param name The name of the shared memory region.
     * 
     * The \p name must be unique (i.e. cannot exist already), otherwise an error occurs.
     * \throws std::system_error on shmp_open fail.
     */
    SharedMemoryPosix(std::string_view name);

    /**
     * \brief Unlink POSIX memory using shm_unlink().
     *
     * shm_unlink is simply called and eventual failures are ignored.
     */
    ~SharedMemoryPosix(void);

    /**
     * \brief Creates a new \ref Segment with the specified size.
     * 
     * \param size The size allocated for the Segment.
     * \return The id the memory Segment has been assigned.
     * \throws std::length_error if the size is 0.
     * 
     * This method simply allocates the requested amount of memory. The Segment's head is at the memory starts position, meaning you can
     * simply write data to the Segment using \ref appendToSegment. To later reserve more space, use \ref increaseSegmentCapacity.
     * \sa Segment.
     * \note The Segment is allocated at \ref m_head and is moved after the segment. The \ref Segment::head is moved to the first data byte's location
     * (after valid length and partial length).
     */
    id_t    createSegment(const size_t size);

    /**
     * \brief Writes data into a Segment starting at the current head of the \ref Segment.
     * 
     * \param id The id of the Segment, returned by e.g. \ref createSegment.
     * \param data A view of the data to be written.
     * \throws std::length_error If appending the data would exceed the Segment's capacity.
     *
     * This method writes data to the Segment and moves the head along with it. No data will be written if
     * the Segment's remaining capacity (total capcaity - head) is too small to fit the entire data.
     */
    void    appendToSegment(id_t id, DataView data);

    /**
     * \brief Writes data into a Segment starting at the specified position.
     * 
     * \param id The id of the Segment, returned by e.g. \ref createSegment.
     * \param position The position within the segment.
     * \param data A view of the data to be written.
     * \throws std::length_error If appending the data would exceed the Segment's capacity.
     *
     * No data will be written if the Segment's remaining capacity (total capcaity - head) is too small to fit the entire data.
     */
    void    writeToSegmentAt(id_t id, position_t position, DataView data);

    /**
     * \brief Increases the total capacity of the Segment.
     * 
     * \param id The id of the segment, returned by e.g. \ref createSegment.
     * \param additionalCapacity The additional amount of memory to be reserved.
     *
     * \note This method will create a new partial segment which might impact performance. If you know the size beforehand, use \ref createSegment with
     * the appropriate size.
     */
    void    increaseSegmentCapacity(id_t id, const size_t additionalCapacity);

    /**
     * \brief Resize a segment to a specific size.
     * 
     * \param id The id of the segment, returned by e.g. \ref createSegment.
     * \param newCapacity The new capacity of the segment.
     *
     * If the new capacity is smaller than the current capacity, all memory exceeding it will be freed and therefore the data lost. Otherwise,
     * behaves like a call to \ref increaseSegmentCapacity with additionalCapacity = newCapacity - currentCapacity
     */
    void    resizeSegment(id_t id, const size_t newCapacity);

    /**
     * \brief Get the current total capacity of the segment.
     * 
     * \param id The id of the segment, returned by e.g. \ref createSegment.
     * \return The Segment's total capacity.
     */
    size_t  getSegmentCapacity(id_t id);

    /**
     * \brief Writes a segment of data into memory.
     * 
     * \param data A view of the data to be written.
     * \return The id the memory segment has been assigned.
     */
    id_t    writeSegment(DataView data);

    /**
     * \brief Deletes a segment of data from memory.
     * 
     * \param id The id the memory segment has been assigned (returned previously by the call to \ref writeSegment).
     */
    // void    deleteSegment(const id_t id);

private:
    /// \brief Increase the size of the shared memory region by some amount.
    void increaseSize(const size_t size = c_size_increase);

    /// \brief Map the shared memory into the virtual address space using mmap().
    void map(void);

    /// \brief Returns the next free id, either the last element in \ref m_unusedIds or \ref m_maxId++.
    id_t getNextId(void);

    /// \brief Extend the segment table in memory by \ref c_contiguous_segment_count.
    void extendSegmentTable(void);

    /// \brief Update the \ref link_t in the segment table for the given id. \todo Possibly change to use Segment.
    void setSegmentTableLink(const id_t id, const link_t link);

    /**
     * \brief Calculate the size of a segment.
     * \param dataSize The size of the data the segment contains.
     * Adds the size of the size_t at the beginning of the segment for partial length encoding
     * as well as the id_t marker and link_t to the next part of the segment.
     */
    inline static constexpr size_t calculatePartialSegmentSize(const size_t dataSize) {
        return dataSize + sizeof(size_t) + sizeof(id_t) + sizeof(link_t);
    }

    /**
     * \brief Helper method to convert a value into big endian.
     * \tparam T Some unsigned integral of size 1, 2, 4 or 8.
     * \param value The value to be endianized.
     * \returns Big endian version of \p value.
     * \todo possibly extract into helper.h or something
     */
    template<std::unsigned_integral T>
    inline static T getBigEndian(T value) {
        static_assert(
            sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
            "Unsupported integer size"
        );

        T endianizedValue;
        if constexpr(sizeof(T) == 1) {
            endianizedValue = value;
        }
        else if constexpr(sizeof(T) == 2) {
            endianizedValue = htobe16(value);
        }
        else if constexpr(sizeof(T) == 4) {
            endianizedValue = htobe32(value);
        }
        else if constexpr(sizeof(T) == 8) {
            endianizedValue = htobe64(value);
        }
        return endianizedValue;
    }

    /**
     * \brief Helper method to write binary data into memory at head. 
     * \tparam T Some unsigned integral of size 1, 2, 4 or 8.
     * \param value The value to be endianized.
     *
     * Increases m_head by sizeof(T).
     */
    template<std::unsigned_integral T>
    inline void write(T value) {
        T endianizedValue = getBigEndian(value);
        std::memcpy(m_memoryStart + m_head, &endianizedValue, sizeof(T));
        m_head += sizeof(T);
    }

    /**
     * \brief Helper method to write binary data into memory at specified position.
     * \tparam T Some unsigned integral of size 1, 2, 4 or 8.
     * \param value The value to be endianized.
     * \param position Offset from \ref m_memoryStart to write to.
     */
    template<std::unsigned_integral T>
    inline void write(T value, uint64_t position) {
        T endianizedValue = getBigEndian(value);
        std::memcpy(m_memoryStart + position, &endianizedValue, sizeof(T));
    }

    /// \brief Constant by which the shared memory's size is increased when more memory is needed (should be the size of one page).
    inline static constexpr size_t      c_size_increase = 4096;

    /// \brief Size of an extension of the segment table.
    inline static constexpr size_t      c_segment_table_size = 
        (c_contiguous_segment_count + 1) * sizeof(id_t) * 2;
    
    static_assert(
        c_segment_table_size < c_size_increase, 
        "This size should not exceed a normal size increase (which should increase by one page), "
        "otherwise please change extendSegmentTable to do more than one increase and delete this assertion "
        "or change it (based on your implementation.)"
    );

    /// \brief Name of the shared memory region.
    std::string                         m_name;

    /// \brief Id returned by shm_open.
    int                                 m_id;

    /// \brief Stores the maximum id that has been assigned to a memory segment.
    id_t                                m_maxId = 0;

    /// \brief Stores unused ids between 0 and \ref m_maxId (can only occur through deletion).
    std::vector<id_t>                   m_unusedIds = {};

    /// \brief Location where the shared memory is mapped into actual RAM (which is a virtual adress).
    std::byte*                          m_memoryStart;

    /// \brief Size of the shared memory region.
    size_t                              m_size = 0;

    /// \brief Head of the shared memory relative to \ref m_memoryStart.
    position_t                          m_head = 0;

    /** 
     * \brief Vector storing Segments where the index correponds to the Segment's id.
     * 
     * The Segment's offsets stored as a partially linked list directly after the version tag. \todo Document using image
     */ 
    std::vector<SegmentInformation>     m_segmentsInformation = {};

    /** 
     * \brief Vector storing the relative position of the partial segment table offsets.
     * 
     * \todo Document using image
     */ 
    std::vector<position_t>             m_segmentTableOffsets = {};
};

}; // namespace memory

}; // namespace UnBARableAI
