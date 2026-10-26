#pragma once
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "memory/shared_memory_types.h"
#include "memory/segment_table.hpp"
#include "serialization/byte_container.h"

#include "serialization/array.tpp"
#include "serialization/optional.tpp"
#include "serialization/partial_segment_table.tpp"
#include "serialization/segment_table_entry.tpp"
#include "serialization/shm_header.tpp"

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
 * \todo Refactor into two classes, one for managing segments and one for the actual posix backend.
 */
class SharedMemoryPosix {
    static_assert(sizeof(std::size_t) == 8, "Invalid bytelength for type 'size_t'");

public:
    /// \brief Constant by which the shared memory's size is increased when more memory is needed
    /// (should be the size of one (or multiple) page(s)).
    inline static constexpr std::size_t c_size_increase = 4096;

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

    /// \brief Copy constructor deleted.
    SharedMemoryPosix(const SharedMemoryPosix&) = delete;

    /// \brief Copy assignment deleted.
    SharedMemoryPosix& operator=(const SharedMemoryPosix&) = delete;

    /// \brief Move constructor.
    SharedMemoryPosix(SharedMemoryPosix&& other);

    /// \brief Move assignment.
    SharedMemoryPosix& operator=(SharedMemoryPosix&& other);

    /**
     * \brief Unlink POSIX memory using shm_unlink().
     *
     * shm_unlink is simply called and eventual failures are ignored.
     */
    ~SharedMemoryPosix(void);

    /**
     * \brief Static method to remove an shm region.
     * \param name Qualified name to remove.
     * \throws On shm_unlink fail.
     *
     * Uses shm_unlink, meaning the name is removed, but existing mapping remain valid until they
     * unmap (using munmap).
     */
    static void remove(std::string_view name);

    /**
     * \brief Update the information about the SharedMemory.
     *
     * \returns Whether or not layout changes have been detected.
     * \note Value changes are not reported, only changes in size.
     */
    bool update(void);

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
    id_t createSegment(const std::size_t size);

    // TODO fix
    inline size_t getSegmentsCount(void) const { return 0; }

    /**
     * \brief Get the segment information, i.e. its position and size in memory.
     */
    inline SegmentInformation getSegmentInformation(id_t segmentId) const {
        return m_segmentTable.getSegmentInformation(segmentId);
    }

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
    void appendToSegment(const id_t segmentId, DataView data);

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
    // void writeToSegmentAt(id_t id, position_t position, DataView data);

    /**
     * \brief Read a segment.
     * \param segmentId Id of the segment to read.
     * \returns Raw data bytes of the segment.
     * \throw std::runtime_error if the segment's id doesnt exist.
     *
     * This method reads all the data bytes (i.e. excluding header and links between partial
     * segments) and copies them into an std::vector<std::byte> which is then returned.
     */
    std::vector<std::byte> readSegment(const id_t segmentId) const;

    /**
     * \brief Increases the total capacity of the Segment.
     *
     * \param id The id of the segment, returned by e.g. \ref createSegment.
     * \param additionalCapacity The additional amount of memory to be reserved.
     *
     * \note This method will create a new partial segment which might impact performance. If you
     * know the size beforehand, use \ref createSegment with the appropriate size.
     */
    // void increaseSegmentCapacity(id_t id, const size_t additionalCapacity);

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
    // void resizeSegment(id_t id, const size_t newCapacity);

    /**
     * \brief Get the current total capacity of the segment.
     *
     * \param id The id of the segment, returned by e.g. \ref createSegment.
     * \return The Segment's total capacity.
     */
    // size_t getSegmentCapacity(id_t id);

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
     * \param segmentId The id the memory segment has been assigned (returned previously by the call
     * to \ref writeSegment).
     */
    void deleteSegment(id_t segmentId);

    /**
     * \brief Get the id of a linked segment.
     * \param segmentId Id of the segment, which links to the wanted segment.
     * \offset Positional offset within in the segment with \p segmentId.
     * \returns Id of the wanted segment or std::nullopt if it couldnt be found.
     * \throws If \p segmentId doesnt even exist.
     *
     * This method returns the id of a segment, that the segment which is passed to the function
     * links to. Starting at the \p offset, the next sizeof(link_t) bytes in the segment (with \p
     * segmentId) are interpreted as a link_t (i.e. the id of another segment). If that id is valid,
     * it returns it. Otherwise returns std::nullopt.
     *
     * \note The link cannot be spread across partial segments, othrwise returns std::nullopt. Also,
     * if the segment hasnt even had data written to it at \p offset (meaning offset +
     * sizeof(link_t) < segInfo.getValidLength()) defaults to returning std::nullopt.
     */
    std::optional<id_t> getLinkedSegment(id_t segmentId, position_t offset) const;

private:
    /// \brief Trivial ctor.
    SharedMemoryPosix(std::string_view name);

    /// \brief Increase the size of the shared memory region by some amount.
    void increaseSize(const size_t size = c_size_increase);

    /**
     * \brief Increases the size only if SegmentTable::getHead() > m_size.
     *
     * The head is managed by the SegmentTable class. Therefore, every time that class allocates
     * memory from the head through operations like SegmentTable::extend() or
     * SegmentTable::allocate() it has to be checked that the head is not larger than the physical
     * memory.
     *
     * This method increases the size by multiples of \ref c_size_increase.
     */
    void increaseSizeIfNecessary(void);

    /**
     * \brief Allocate a new segment with \p dataSize.
     * \param dataSize Number of data bytes for the new segment.
     *
     * Allocates a new segment in the SegmentTable and then updates the link and increases the size
     * if necessary.
     *
     * \see updateSegmentTableLink, increaseSizeIfNecessary.
     */
    const SegmentInformation& allocateNewSegment(const std::size_t dataSize);

    /// \brief Map the shared memory into the virtual address space using mmap().
    void map(void);

    /// \brief Release acquired ressources.
    void release(void);

    /// \brief Extend m_segmentTable and write the new PartialSegmentTable to memory.
    void extendSegmentTable(void);

    /**
     * \brief Updates the SegmentTable's link.
     * \param segmentId Id of the segment.
     * \param link Link for the segment.
     *
     * \note This method relies on the fact that SegmentTable::allocate only affects the link for
     * the segment with id segmentId in the PartialSegmentTable, nothing else.
     */
    void updateSegmentTableLink(const id_t segmentId, const std::optional<link_t> link);

    /**
     * \brief Update the valid length as written in shm.
     * \param segmentId Id of the segment to update.
     * \throws If segmentId is unknown / uninitialized.
     * \note This method requires that only the head changed when SegmentTable::advanceHead is
     * called. If this isnt ensured, the method breaks.
     */
    void updateValidLength(const id_t segmentId);

    /**
     * \brief Read raw data bytes.
     * \param position Position to start reading from.
     * \param numBytes Number of bytes to read.
     * \returns A std::vector<std::byte> containing the read data.
     */
    std::vector<std::byte> read(position_t position, size_t numBytes) const;

    /**
     * \brief Helper method to write any trivial serializable into memory at \p position.
     * \tparam T Any TrivialSerializable.
     * \param value The value to be written.
     * \param position Offset from \ref m_memoryStart to write to.
     */
    template <serialization::TrivialSerializable T>
    inline void write(T value, uint64_t position) {
        auto serialized = serialization::serialize(value);
        static_assert(serialization::ByteContainer<std::remove_cvref_t<decltype(serialized)>>,
                      "T's SerializeInformation does not return a ByteContainer on serialization.");
        std::memcpy(m_memoryStart + position, serialized.data(), serialized.size());
    }

    /// \brief Checks that magic and version are valid at the start of the shm.
    bool isStartValid(void) const;

    // static_assert(
    //     c_segment_table_size < c_size_increase,
    //     "This size should not exceed a normal size increase (which should increase by one page),
    //     " "otherwise please change extendSegmentTable to do more than one increase and delete
    //     this " "assertion " "or change it (based on your implementation.)");

    /// \brief Name of the shared memory region.
    std::string m_name;

    /// \brief Id returned by shm_open.
    int m_id = -1;

    /// \brief Location where the shared memory is mapped into actual RAM (which is a virtual
    /// adress).
    std::byte* m_memoryStart = nullptr;

    /// \brief Size of the shared memory region.
    size_t m_size = 0;

    /// \brief Handles segments.
    SegmentTable m_segmentTable = SegmentTable();
};

};  // namespace memory

};  // namespace UnBARableAINS
