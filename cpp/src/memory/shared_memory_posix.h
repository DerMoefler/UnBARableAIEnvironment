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

namespace UnBARableAI {

namespace memory {

/**
 * \brief Uses POSIX shared memory.
 * \todo fix use of uint64_t and id_t to refer to an index / size type.
 * People seem to prefer POSIX shared memory over system V's, which is used by \ref SharedMemoryUnix.
 */
class SharedMemoryPosix {
public:
    static_assert(sizeof(id_t) == 4, "Invalid size for type 'id_t', has to be 4 bytes.");
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
     * \brief Writes a segment of data into memory.
     * 
     * \return The id the memory segment has been assigned internally.
     */
    id_t    writeSegment(std::span<const std::byte> data);

    /**
     * \brief Deletes a segment of data from memory.
     * 
     * \param id The id the memory segment has been assigned internally.
     */
    // void    deleteSegment(const id_t id);

private:
    /// \brief Increase the size of the shared memory region by some amount.
    void increaseSize(uint64_t size = c_size_increase);

    /// \brief Map the shared memory into the virtual address space using mmap().
    void map(void);

    /// \brief Returns the next free id, either the last element in \ref m_unusedIds or \ref m_maxId++.
    id_t getNextId(void);

    /// \brief Extend the segment table in memory by \ref c_contiguous_segment_count.
    void extendSegmentTable(void);

    /// \brief Update the "link" (i.e. offset from \ref m_memoryStart) in the segment table for the given id.
    void setSegmentTableLink(id_t id, uint64_t position);

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
    inline static constexpr uint64_t    c_size_increase = 4096;

    /// \brief Size of an extension of the segment table.
    inline static constexpr uint64_t    c_segment_table_size = 
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
    uint64_t                            m_size = 0;

    /// \brief Head of the shared memory relative to \ref m_memoryStart.
    uint64_t                            m_head = 0;

    /** 
     * \brief Table storing entries [offset], i.e. the offset from the start of shared memory to the segment identified by id, which corresponds to the index.
     * 
     * This list is stored as a partially linked list directly after the version tag. \todo Document using image
     */ 
    std::vector<uint64_t>               m_segmentOffsets = {};

    /** 
     * \brief Vector storing the relative position of the partial segment table offsets.
     * 
     * \todo Document using image
     */ 
    std::vector<uint64_t>               m_segmentTableOffsets = {};


};

}; // namespace memory

}; // namespace UnBARableAI
