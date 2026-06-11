#include "shared_memory_posix.h"
#include "memory/shared_memory_impl.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <system_error>

namespace UnBARableAINS {

namespace memory {

SharedMemoryPosix::SharedMemoryPosix(std::string_view name) 
    : m_name(name)
{
    m_id = shm_open(m_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
    if(m_id == -1) {
        throw std::system_error(errno, std::generic_category(), "shm_open failed");
    }
    increaseSize();
    write(c_UnBARableAI_magic);
    write(c_version);
    extendSegmentTable();
}

SharedMemoryPosix::~SharedMemoryPosix(void) {
    shm_unlink(m_name.c_str());
}

id_t SharedMemoryPosix::writeSegment(std::span<const std::byte> data) {
    id_t segmentId = getNextId();
    if(segmentId >= m_segmentOffsets.capacity() - 1) {
        extendSegmentTable();
    }
    // Get size and extend shm if needed
    const size_t dataSize = data.size();
    const size_t segmentSize = calcualteSegmentSize(dataSize);
    if((m_head + segmentSize) > m_size) {
        // Increase by n * c_size_increase to fit the data into memory.
        const size_t sizeIncrease = (segmentSize / c_size_increase + 1) * c_size_increase;
        increaseSize(sizeIncrease);
    }
    // Layout: (segmentSize, data[0], data[1], ..., data[dataSize], segment_link_id, link)
    write(segmentSize);
    setSegmentTableLink(segmentId, m_head);
    for(size_t i = 0; i < dataSize; i++) {
        write(static_cast<uint8_t>(data[i]));
    }
    write(c_segment_link_id);
    write(static_cast<link_t>(0x00));
    
    return segmentId;
}

void SharedMemoryPosix::increaseSize(const size_t size) {
    ftruncate(m_id, m_size + size);
    m_size += size;
    map();
}

void SharedMemoryPosix::map(void) {
    m_memoryStart = static_cast<std::byte*>(mmap(NULL, m_size, PROT_READ | PROT_WRITE, MAP_SHARED_VALIDATE, m_id, 0));
}

id_t SharedMemoryPosix::getNextId(void) {
    id_t id;
    if (m_unusedIds.empty()) {
        id = m_maxId++;
    }
    else {
        id = m_unusedIds.back();
        m_unusedIds.pop_back();
    }
    return id;
}

void SharedMemoryPosix::extendSegmentTable(void) {
    if((m_head + c_segment_table_size) > m_size) {
        increaseSize();
    }

    id_t partialSegmentTablesCount = m_segmentTableOffsets.size() ;
    // Link this new partial segment table to previous one.
    if (partialSegmentTablesCount) {
        // Start of the previous partial table
        id_t previousOffset = m_segmentTableOffsets.back();
        // Write offset for the current head in memory
        id_t writePositionPreviousOffset = previousOffset + c_segment_table_size - sizeof(id_t);
        write(static_cast<link_t>(m_head), writePositionPreviousOffset);
    }

    m_segmentOffsets.reserve(c_contiguous_segment_count + 1);
    id_t firstId = partialSegmentTablesCount * c_contiguous_segment_count;
    m_segmentTableOffsets.push_back(m_head);
    for(id_t i = firstId; i < firstId + c_contiguous_segment_count; i++) {
        write(i);
        write(static_cast<link_t>(0x00));
    }
    // Create last entry as a link, which will be overriden by the next call to this method as per above
    write(c_partial_table_link_id);
    write(static_cast<link_t>(0x00));
}

void SharedMemoryPosix::setSegmentTableLink(const id_t id, const link_t position) {
    size_t partialTableIdx = id / c_contiguous_segment_count;
    id_t partialTableStart = m_segmentTableOffsets[partialTableIdx];
    id_t writeOffset =  partialTableStart 
                      + (id % c_contiguous_segment_count) * 2 * sizeof(id_t) 
                      + sizeof(id_t);
    write(position, writeOffset);
}

}; // namespace memory

}; // namespace UnBARableAI
