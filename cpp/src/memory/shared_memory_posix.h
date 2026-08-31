#pragma once
#include <endian.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "../memory/shared_memory_types.h"
#include "id/id_allocator.hpp"
#include "partially_linked_list_information.h"

namespace UnBARableAINS {

namespace memory {

/**
 * \brief Implements a \ref SharedMemoryImpl using POSIX shared memory.
 * \todo fix use of uint64_t and id_t to refer to an index / size type.
 *
 * \todo document
 * \image html shared_memory_layout.svg
 * \todo There seems to be an out of bounds access problem when using c_contiguous_segments = 2 in
 * shared_memory_test
 *
 * People seem to prefer POSIX shared memory over system V's, which is used by \ref
 * SharedMemoryUnix.
 */
class SharedMemoryPosix {
    static_assert(sizeof(size_t) == 8, "Invalid bytelength for type 'size_t'");

public:
    /// \brief Alias for a view to some data (e.g. to write to a segment).
    using DataView = const std::span<const std::byte>;

    /**
     * \brief Hold information about a Segment in shared memory.
     *
     * A instance of SegmentInformation can either be constructed as or set to be invalid. This
     * means that the Segment does not actually exist, meaning no information about it can be
     * queried. This is to allow reserving space for a SegmentInformation object without it holding
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

        id_t m_id;
        bool m_valid;
        PartiallyLinkedListInformation m_information;
    };

    /// \brief The id used to signal that the next entry is not an offset for a data segment but
    /// rather the offset to the next partial segment table.
    inline static constexpr id_t c_partial_table_link_id = 0xFE'DC'BA'98;  //'76'54'32'10;

    /**
     * \brief Number of contiguous segments in the partially linked list.
     * The last element is a relative pointer to next array. \todo image
     */
    inline static constexpr uint32_t c_contiguous_segment_count = 32;

    /// \brief The id used to signal that the following bytes compose a \ref position_t to where the
    /// segment continues.
    inline static constexpr id_t c_segment_link_id = 0xAA'AA'AA'AA;

    /**
     * \brief Create shared memory using shm_open().
     * \param name Name of the shared memory region to create.
     * \returns SharedMemoryPosix object to access the shm region.
     *
     * The \p name must be unique (i.e. cannot exist already), otherwise an error occurs.
     * \throws std::system_error on shm_open fail.
     */
    static SharedMemoryPosix create(std::string_view name);

    /**
     * \brief Open an existing shared memory region using shm_open().
     * \param Name of the exisiting shared memory region to open.
     * \returns SharedMemoryPosix object to access the shm region.
     *
     * A region with the \p name must already exist, otherwise an error occurs.
     * \throws std::system_error on shm_open fail.
     */
    static SharedMemoryPosix open(std::string_view name);

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
     * This method simply allocates the requested amount of memory. The Segment's head is at the
     * memory starts position, meaning you can simply write data to the Segment using \ref
     * appendToSegment. To later reserve more space, use \ref increaseSegmentCapacity.
     * \sa Segment.
     * \note The Segment is allocated at \ref m_head and is moved after the segment. The \ref
     * Segment::head is moved to the first data byte's location (after valid length and partial
     * length).
     */
    id_t createSegment(const size_t size);

    /**
     * \brief Writes data into a Segment starting at the current head of the \ref Segment.
     *
     * \param id The id of the Segment, returned by e.g. \ref createSegment.
     * \param data A view of the data to be written.
     * \throws std::length_error If appending the data would exceed the Segment's capacity.
     *
     * This method writes data to the Segment and moves the head along with it. No data will be
     * written if the Segment's remaining capacity (total capcaity - head) is too small to fit the
     * entire data.
     */
    void appendToSegment(const id_t id, DataView data);

    /**
     * \brief Writes data into a Segment starting at the specified position.
     *
     * \param id The id of the Segment, returned by e.g. \ref createSegment.
     * \param position The position within the segment.
     * \param data A view of the data to be written.
     * \throws std::length_error If appending the data would exceed the Segment's capacity.
     *
     * No data will be written if the Segment's remaining capacity (total capcaity - head) is too
     * small to fit the entire data.
     */
    void writeToSegmentAt(id_t id, position_t position, DataView data);

    /**
     * \brief Increases the total capacity of the Segment.
     *
     * \param id The id of the segment, returned by e.g. \ref createSegment.
     * \param additionalCapacity The additional amount of memory to be reserved.
     *
     * \note This method will create a new partial segment which might impact performance. If you
     * know the size beforehand, use \ref createSegment with the appropriate size.
     */
    void increaseSegmentCapacity(id_t id, const size_t additionalCapacity);

    /**
     * \brief Resize a segment to a specific size.
     *
     * \param id The id of the segment, returned by e.g. \ref createSegment.
     * \param newCapacity The new capacity of the segment.
     *
     * If the new capacity is smaller than the current capacity, all memory exceeding it will be
     * freed and therefore the data lost. Otherwise, behaves like a call to \ref
     * increaseSegmentCapacity with additionalCapacity = newCapacity - currentCapacity
     */
    void resizeSegment(id_t id, const size_t newCapacity);

    /**
     * \brief Get the current total capacity of the segment.
     *
     * \param id The id of the segment, returned by e.g. \ref createSegment.
     * \return The Segment's total capacity.
     */
    size_t getSegmentCapacity(id_t id);

    /**
     * \brief Writes a segment of data into memory.
     *
     * \param data A view of the data to be written.
     * \return The id the memory segment has been assigned.
     */
    id_t writeSegment(DataView data);

    /**
     * \brief Deletes a segment of data from memory.
     *
     * \param id The id the memory segment has been assigned (returned previously by the call to
     * \ref writeSegment).
     */
    // void    deleteSegment(const id_t id);

private:
    /// \brief Trivial ctor.
    SharedMemoryPosix(std::string_view name);
    /**
     * \brief Finds a segmentInformation for the id or throws if it doesnt exist.
     */
    SegmentInformation& findSegmentInformation(id_t segmentId);

    /// \brief Const overload for findSegmentInformation.
    inline const SegmentInformation& findSegmentInformation(id_t segmentId) const {
        return findSegmentInformation(segmentId);
    }

    /**
     * \brief Returns the associated SegmentInformation's index in m_segmentsInformation for the
     * specified id, if present. Otherwise, returns std::nullopt.
     */
    std::optional<size_t> findSegmentInformationIndex(id_t segmentId) const;

    /// \brief Increase the size of the shared memory region by some amount.
    void increaseSize(const size_t size = c_size_increase);

    /// \brief Map the shared memory into the virtual address space using mmap().
    void map(void);

    /// \brief Extend the segment table in memory by \ref c_contiguous_segment_count.
    void extendSegmentTable(void);

    /// \brief Update the \ref link_t in the segment table for the given id. \todo Possibly change
    /// to use Segment.
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
    template <std::unsigned_integral T>
    inline static T getBigEndian(T value) {
        static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                      "Unsupported integer size");

        T endianizedValue;
        if constexpr (sizeof(T) == 1) {
            endianizedValue = value;
        }
        else if constexpr (sizeof(T) == 2) {
            endianizedValue = htobe16(value);
        }
        else if constexpr (sizeof(T) == 4) {
            endianizedValue = htobe32(value);
        }
        else if constexpr (sizeof(T) == 8) {
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
    template <std::unsigned_integral T>
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
    template <std::unsigned_integral T>
    inline void write(T value, uint64_t position) {
        T endianizedValue = getBigEndian(value);
        std::memcpy(m_memoryStart + position, &endianizedValue, sizeof(T));
    }

    /// \brief Constant by which the shared memory's size is increased when more memory is needed
    /// (should be the size of one page).
    inline static constexpr size_t c_size_increase = 4096;

    /// \brief Size of an extension of the segment table.
    inline static constexpr size_t c_segment_table_size =
        (c_contiguous_segment_count + 1) * sizeof(id_t) * 2;

    static_assert(
        c_segment_table_size < c_size_increase,
        "This size should not exceed a normal size increase (which should increase by one page), "
        "otherwise please change extendSegmentTable to do more than one increase and delete this "
        "assertion "
        "or change it (based on your implementation.)");

    /// \brief Name of the shared memory region.
    std::string m_name;

    /// \brief Id returned by shm_open.
    int m_id;

    /// \brief Handles id allocation for segments.
    id::IdAllocator m_idAllocator;

    /// \brief Location where the shared memory is mapped into actual RAM (which is a virtual
    /// adress).
    std::byte* m_memoryStart;

    /// \brief Size of the shared memory region.
    size_t m_size = 0;

    /// \brief Head of the shared memory relative to \ref m_memoryStart.
    position_t m_head = 0;

    /**
     * \brief Vector storing Segments where the index correponds to the Segment's id.
     *
     * The Segment's offsets stored as a partially linked list directly after the version tag. \todo
     * Document using image
     */
    std::vector<SegmentInformation> m_segmentsInformation = {};

    /**
     * \brief Vector storing the relative position of the partial segment table offsets.
     *
     * \todo Document using image
     */
    std::vector<position_t> m_segmentTableOffsets = {};
};

};  // namespace memory

};  // namespace UnBARableAINS
