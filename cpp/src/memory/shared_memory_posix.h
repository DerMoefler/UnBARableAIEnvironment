#pragma once
#include <cstddef>
#include <string>
#include <cstdint>
#include <vector>
#include <span>
#include <cstring>
#include <concepts>
#include <endian.h>

#include "memory/shared_memory_impl.h"

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
    /** 
    * \brief Type alias for a 'link', which is specifically a relative pointer linking objects together, e.g. the segment table or a segment itself.
    *
    * \note Raw pointers are not usable in shared memory, which is why this class uses offsets
    * relative to \ref m_memoryStart.
    */ 
    using link_t = id_t;

    /** 
    * \brief Type alias for a specific position in the shared memory.
    *
    * The position starts at 0 from the \ref m_memoryStart. Conceptually very similar to \ref link_t. The difference is, that
    * a \ref link_t is specific to linking binary objects like the segment table or segments themselves together. A position is effectively
    * the general term, while \ref link_t is a semantically special position_t.
    */ 
    using position_t = id_t;

    /// \brief Alias for a view to some data (e.g. to write to a segment).
    using DataView = const std::span<const std::byte>;

    /**
     * \brief A struct to hold information about the memory layout of a segment in shared memory.
     * 
     * In memory, a Segment's \ref id is stored in a partially linked table with its associated \ref start. A Segment itself is also partially
     * linked. The first "partial" Segment, i.e. the position in memory that \ref start refers to, starts with the current length of the entire
     * segment, i.e. the amount of valid data. Then, each partial segment (including the first) stores its own length at the beginning (or in the case
     * of the first segment, after the valid length). This is so the end of the partial segment can be determined. The last few bytes of the Segment
     * denote a \ref link_t to the following partial segment. The value of this \ref link_t is 0 if this is the last segment.
     * \note The valid length as well as the partial segment length both include themselves. The link_t is included in the partial segment size as well.
     * \note The first segment's partial length excldues the valid length while the valid length includes the partial length.
     * \note It is not guaranteed for the head to be in the last partial segment. The user may allocate multiple partial Segments without ever actually
     * writing data to it. 
     */
    struct Segment {
        /// \brief A boolean determining whether the instance holds information about an actual segment or not.
        bool                        valid = false;
        /// \brief The id of the segment.
        id_t                        id;
        /// \brief The start of the segment relative to the shared memory's start.
        position_t                  start;  
        /// \brief The head of the segment relative to the shared memory's start
        position_t                  head;
        /// \brief The total capacity of the segment.
        size_t                      size;
        /// \brief The offsets of partial segments relative to the shared memory's start.
        std::vector<position_t>     offsets;
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
    std::vector<Segment>                m_segments = {};

    /** 
     * \brief Vector storing the relative position of the partial segment table offsets.
     * 
     * \todo Document using image
     */ 
    std::vector<position_t>             m_segmentTableOffsets = {};
};

}; // namespace memory

}; // namespace UnBARableAI
