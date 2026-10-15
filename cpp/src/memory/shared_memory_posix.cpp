#include "shared_memory_posix.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "serialization/serialize_information.h"

#ifdef __APPLE__
#define UNBAR_MMAP_FLAG MAP_SHARED
#else
#define UNBAR_MMAP_FLAG MAP_SHARED_VALIDATE
#endif

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <system_error>
#include <utility>

#include "memory/shared_memory_types.h"
#include "utility/unsigned_of_size.hpp"

static auto logger = spdlog::stdout_color_mt("SharedMemoryPosix");

namespace UnBARableAINS {

namespace memory {

template <typename S>
using SI = serialization::SerializeInformation<S>;

namespace detail {

template <std::size_t numBytes>
auto deserializeUnsigned(const std::vector<std::byte>& serialized)
    -> util::UnsignedOfSize_T<numBytes> {
    using Unsigned = util::UnsignedOfSize_T<numBytes>;
    assert(serialized.size() == numBytes &&
           "Vector's size must match (compile-time) request number of bytes");
    auto view = std::span<const std::byte, numBytes>{serialized.data(), numBytes};
    return SI<Unsigned>::deserialize(view);
}

}  // namespace detail

SharedMemoryPosix SharedMemoryPosix::create(std::string_view name) {
    SharedMemoryPosix vThis(name);
    vThis.m_id = shm_open(vThis.m_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
    if (vThis.m_id == -1) {
        throw std::system_error(errno, std::generic_category(), "shm_open failed");
    }
    vThis.increaseSize();
    vThis.write(c_UnBARableAI_magic);
    vThis.write(c_version);
    vThis.extendSegmentTable();
    SPDLOG_LOGGER_INFO(logger, "Created shared memory \"{}\"", name);
    return vThis;
}

SharedMemoryPosix SharedMemoryPosix::open(std::string_view name) {
    SharedMemoryPosix vThis(name);
    vThis.m_id = shm_open(vThis.m_name.c_str(), O_RDWR, 0600);
    if (vThis.m_id == -1) {
        throw std::system_error(errno, std::generic_category(), "shm_open failed");
    }

    vThis.update();

    SPDLOG_LOGGER_INFO(logger, "Opened shared memory \"{}\"", name);
    return vThis;
}

SharedMemoryPosix::SharedMemoryPosix(SharedMemoryPosix&& other)
    : m_name(std::move(other.m_name))
    , m_id(std::exchange(other.m_id, -1))
    , m_idAllocator(std::move(other.m_idAllocator))
    , m_memoryStart(std::exchange(other.m_memoryStart, nullptr))
    , m_size(std::exchange(other.m_size, 0))
    , m_head(std::exchange(other.m_head, 0))
    , m_segmentsInformation(std::move(other.m_segmentsInformation))
    , m_segmentTableOffsets(std::move(other.m_segmentTableOffsets)) {}

SharedMemoryPosix& SharedMemoryPosix::operator=(SharedMemoryPosix&& other) {
    if (this == &other) {
        return *this;
    }

    release();

    m_name = std::move(other.m_name);
    m_id = std::exchange(other.m_id, -1);
    m_idAllocator = id::IdAllocator();
    m_memoryStart = std::exchange(other.m_memoryStart, nullptr);
    m_size = std::exchange(other.m_size, 0);
    m_head = std::exchange(other.m_head, 0);
    m_segmentsInformation = std::move(other.m_segmentsInformation);
    m_segmentTableOffsets = std::move(other.m_segmentTableOffsets);

    return *this;
}

SharedMemoryPosix::~SharedMemoryPosix(void) { release(); }

void SharedMemoryPosix::remove(std::string_view name) {
    std::string nullTerminatedName(name);
    if (shm_unlink(nullTerminatedName.c_str()) == -1) {
        SPDLOG_LOGGER_ERROR(logger, "shm_unlink failed");
        throw std::system_error(errno, std::generic_category(), "shm_unlink failed");
    }
    SPDLOG_LOGGER_INFO(logger, "Removed shared memory \"{}\"", name);
}

bool SharedMemoryPosix::update(void) {
    bool change = false;
    struct stat info{};
    if (fstat(m_id, &info) == -1) {
        SPDLOG_LOGGER_ERROR(logger, "fstat failed");
        throw std::system_error(errno, std::generic_category(), "fstat failed");
    }
    const auto oldSize = m_size;
    m_size = info.st_size;

    change |= oldSize != m_size;
    if (change) {
        map();
    }

    if (!isStartValid()) throw std::runtime_error("Shared memory's start is invalid.");
    if (!isSegmentTableValid())
        throw std::runtime_error("Shared memory's segment table is invalid.");

    const auto oldSegmentsInformation = m_segmentsInformation;
    if (!initializeSegmentsInformation())
        throw std::runtime_error("Initializing segments information failed");

    if (oldSegmentsInformation != m_segmentsInformation) {
        change = true;
        initializeIdAllocator();
    }

    updateHead();

    SPDLOG_LOGGER_DEBUG(logger, "Updated shared memory \"\"", m_name);
    return change;
}

id_t SharedMemoryPosix::createSegment(const size_t size) {
    // std::cout << "SharedMemoryPosix::createSegment: Requested size = " << size << std::endl;
    if (!size) {
        throw std::length_error("Don't dare create an empty segment.");
    }
    id_t newSegmentId = m_idAllocator.allocate();
    if (newSegmentId >= m_segmentsInformation.capacity() - 1) {
        extendSegmentTable();
    }
    // std::cout << "Trying to create segment with id " << newSegmentId << std::endl;
    // Initialize segment information
    SegmentInformation& sInformation = m_segmentsInformation[newSegmentId];
    sInformation.initialize(m_head, size);

    // extend shm if needed (an additional size_t is needed for the encoding of the valid length in
    // the beginning)
    const size_t segmentSize = sInformation.getSize();
    if ((m_head + segmentSize) > m_size) {
        // Increase by n * c_size_increase to fit the data into memory.
        const size_t sizeIncrease = (segmentSize / c_size_increase + 1) * c_size_increase;
        increaseSize(sizeIncrease);
    }
    // TODO check if works and document somewhere
    // Layout (first segment): (validLength, partialLength (not valid but total, also including
    // validLength field), data[0], data[1], ..., data[dataSize], segment_link_id, link)
    write(static_cast<size_t>(0), m_head);
    write(segmentSize, m_head + sizeof(size_t));
    setSegmentTableLink(newSegmentId, m_head);
    write(static_cast<link_t>(0x00), m_head + segmentSize - sizeof(link_t));

    // Move head to after the segment
    m_head += segmentSize;
    // std::cout << "SharedMemoryPosix::createSegment: Successfully created segment." << std::endl;

    SPDLOG_LOGGER_DEBUG(logger, "New segment with id={} created.", newSegmentId);
    return newSegmentId;
}

void SharedMemoryPosix::appendToSegment(const id_t id, DataView data) {
    // std::cout << "SharedMemoryPosix::apendToSegment" << std::endl;
    SegmentInformation& segmentInformation = findSegmentInformation(id);
    for (size_t i = 0; i < data.size(); i++) {
        write(static_cast<uint8_t>(data[i]), segmentInformation.getHead());
        segmentInformation.advanceHead(1);
    }
    updateValidLength(id);
}

std::vector<std::byte> SharedMemoryPosix::readSegment(id_t segmentId) const {
    const SegmentInformation& segmentInformation = findSegmentInformation(segmentId);
    size_t dataSize = segmentInformation.getValidLength();
    size_t readDataSize = 0;

    std::vector<std::byte> rawSegmentData(dataSize);

    for (int i = 0; i < segmentInformation.getPartialSegmentsCount(); i++) {
        position_t partialSegmentDataStart = segmentInformation.getDataStart(i);
        std::size_t partialSegmentDataSize = segmentInformation.getDataSize(i);

        std::size_t remainingDataSize = dataSize - readDataSize;
        std::size_t numBytesToRead = std::min(partialSegmentDataSize, remainingDataSize);

        std::memcpy(rawSegmentData.data() + readDataSize, m_memoryStart + partialSegmentDataStart,
                    numBytesToRead);
    }

    return rawSegmentData;
}

id_t SharedMemoryPosix::writeSegment(DataView data) {
    id_t segmentId = createSegment(data.size());
    appendToSegment(segmentId, data);
    return segmentId;
}

void SharedMemoryPosix::deleteSegment(id_t segmentId) {
    auto& si = findSegmentInformation(segmentId);
    si.invalidate();
    m_idAllocator.release(segmentId);
    updateHead();
}

std::optional<id_t> SharedMemoryPosix::getLinkedSegment(id_t segmentId, position_t offset) const {
    const SegmentInformation& segmentInformation = findSegmentInformation(segmentId);
    // Link cannot be valid if data hasnt even been written
    if (offset + sizeof(link_t) > segmentInformation.getValidLength()) {
        return std::nullopt;
    }
    if (!segmentInformation.isSequentialReadSafe(offset, sizeof(link_t))) {
        return std::nullopt;
    }
    // TODO readSegmentAt function or something
    position_t dataPosition = segmentInformation.getDataPosition(offset);
    auto serialized = read(dataPosition, sizeof(link_t));
    assert(serialized.size() == sizeof(link_t) && "Link's size must be sizeof(link_t) bytes");
    auto linkedSegmentId = SI<link_t>::deserialize(
        std::span<const std::byte, sizeof(link_t)>{serialized.data(), sizeof(link_t)});
    const SegmentInformation& linkedSegmentInformation = findSegmentInformation(linkedSegmentId);
    if (linkedSegmentInformation.isValid()) {
        return linkedSegmentId;
    }
    else {
        return std::nullopt;
    }
}

SharedMemoryPosix::SharedMemoryPosix(std::string_view name)
    : m_name(name) {}

// TODO refactor into unified implementation for mutable and immutable
auto SharedMemoryPosix::findSegmentInformation(id_t segmentId) const -> const SegmentInformation& {
    auto indexOpt = findSegmentInformationIndex(segmentId);
    if (!indexOpt.has_value()) {
        throw std::runtime_error("No segment with that id");
    }
    const SegmentInformation& segmentInformation = m_segmentsInformation[indexOpt.value()];
    return segmentInformation;
}

auto SharedMemoryPosix::findSegmentInformation(id_t segmentId) -> SegmentInformation& {
    auto indexOpt = findSegmentInformationIndex(segmentId);
    if (!indexOpt.has_value()) {
        throw std::runtime_error("No segment with that id");
    }
    SegmentInformation& segmentInformation = m_segmentsInformation[indexOpt.value()];
    return segmentInformation;
}

std::optional<size_t> SharedMemoryPosix::findSegmentInformationIndex(id_t segmentId) const {
    auto it = std::find_if(m_segmentsInformation.begin(), m_segmentsInformation.end(),
                           [segmentId](const SegmentInformation& segInfo) -> bool {
                               return (segInfo.getId() == segmentId) && segInfo.isValid();
                           });
    if (it == m_segmentsInformation.end()) {
        return std::nullopt;
    }
    else {
        return std::distance(m_segmentsInformation.begin(), it);
    }
}

void SharedMemoryPosix::increaseSize(const size_t size) {
    ftruncate(m_id, m_size + size);
    m_size += size;
    map();
}

void SharedMemoryPosix::map(void) {
    SPDLOG_LOGGER_TRACE(logger, "Mapping region into address space");
    m_memoryStart = static_cast<std::byte*>(
        mmap(NULL, m_size, PROT_READ | PROT_WRITE, UNBAR_MMAP_FLAG, m_id, 0));
}

void SharedMemoryPosix::release(void) {
    if (m_memoryStart != nullptr) {
        // Ignores errors
        munmap(m_memoryStart, m_size);
        m_memoryStart = nullptr;
    }
    if (m_id != -1) {
        close(m_id);
        m_id = -1;
    }

    m_idAllocator = id::IdAllocator();
    m_name = {};
    m_size = 0;
    m_head = 0;
    m_segmentsInformation = {};
    m_segmentTableOffsets = {};
}

bool SharedMemoryPosix::initializeSegmentsInformation(void) {
    if (!m_segmentsInformation.empty() || !m_segmentTableOffsets.empty()) {
        return false;
    }

    size_t partialSegmentTableStart = c_segment_table_start;
    constexpr size_t c_entry_size = sizeof(id_t) + sizeof(link_t);

    // Returns whether another partial segment table follows
    auto initializePartialSegmentsInformation = [&]() -> bool {
        // Add this to offsets
        m_segmentTableOffsets.push_back(partialSegmentTableStart);

        m_segmentsInformation.reserve(c_segment_table_start);
        // Update head
        m_head = std::max(m_head,
                          static_cast<position_t>(partialSegmentTableStart + c_segment_table_size));

        std::vector<std::byte> partialSegmentTable =
            read(partialSegmentTableStart, c_segment_table_size);
        assert(partialSegmentTable.size() == c_segment_table_size &&
               "Segment table size must satisfy compile time definition");
        std::size_t offset;
        for (int i = 0; i < c_contiguous_segment_count; i++) {
            offset = i * c_entry_size;
            auto serializedId = std::span<const std::byte, sizeof(id_t)>{
                partialSegmentTable.data() + offset, sizeof(id_t)};
            auto serializedLink = std::span<const std::byte, sizeof(id_t)>{
                partialSegmentTable.data() + offset + sizeof(id_t), sizeof(id_t)};
            auto id = SI<id_t>::deserialize(serializedId);
            auto link = SI<id_t>::deserialize(serializedLink);

            if (!link) {
                SPDLOG_LOGGER_TRACE(logger, "Segment with id {} is empty", id);
                m_segmentsInformation.push_back(SegmentInformation(id));
                continue;
            }

            size_t validLength =
                detail::deserializeUnsigned<sizeof(size_t)>(read(link, sizeof(size_t)));
            size_t partialSegmentLength = detail::deserializeUnsigned<sizeof(size_t)>(
                read(link + sizeof(size_t), sizeof(size_t)));
            SPDLOG_LOGGER_TRACE(
                logger,
                "Segment with id {} has {} paylaod bytes. First partial segment size is {} bytes",
                id, validLength, partialSegmentLength);
            // std::cout << "SharedMemoryPosix::initializeSegmentsInformation (Id: " << id
            //           << ", Link:" << link << ", ValidLength: " << validLength << ")\n";
            SegmentInformation si(id, link,
                                  partialSegmentLength - 2 * sizeof(size_t) - sizeof(link_t));
            si.advanceHead(validLength);
            link_t nextPartialSegment = detail::deserializeUnsigned<sizeof(link_t)>(
                read(link + partialSegmentLength - sizeof(link_t), sizeof(link_t)));
            if (nextPartialSegment > 0) {
                throw std::runtime_error(
                    "More than one partial segment currently unsupported by "
                    "SharedMemoryPosix::initializeSegmentsInformation");
            }
            m_segmentsInformation.push_back(std::move(si));
        }
        position_t continuation = SI<position_t>::deserialize(
            std::span{partialSegmentTable}
                .subspan<c_segment_table_size - sizeof(link_t), sizeof(link_t)>());
        partialSegmentTableStart = continuation;
        return continuation != 0;
    };

    auto run = true;
    while (run) {
        run = initializePartialSegmentsInformation();
    }

    return true;
}

void SharedMemoryPosix::updateHead(void) {
    // Minimum head position
    m_head = c_segment_table_start;

    for (const auto tableOffset : m_segmentTableOffsets) {
        m_head = std::max(m_head, static_cast<position_t>(tableOffset + c_segment_table_size));
    }
    for (const auto& si : m_segmentsInformation) {
        if (!si.isValid()) continue;
        m_head = std::max(m_head, si.getMemoryEnd());
    }
}

void SharedMemoryPosix::initializeIdAllocator(void) {
    m_idAllocator = id::IdAllocator();
    std::set<id::id_t> usedIds{};
    for (std::size_t i = 0; i < m_segmentsInformation.size(); i++) {
        if (m_segmentsInformation[i].isValid()) {
            usedIds.insert(static_cast<id::id_t>(i));
        }
    }
    m_idAllocator.setUsedIds(usedIds);
}

void SharedMemoryPosix::extendSegmentTable(void) {
    if ((m_head + c_segment_table_size) > m_size) {
        increaseSize();
    }

    id_t partialSegmentTablesCount = m_segmentTableOffsets.size();
    // Link this new partial segment table to previous one.
    if (partialSegmentTablesCount) {
        // Start of the previous partial table
        id_t previousOffset = m_segmentTableOffsets.back();
        // Write offset for the current head in memory
        id_t writePositionPreviousOffset = previousOffset + c_segment_table_size - sizeof(id_t);
        write(static_cast<link_t>(m_head), writePositionPreviousOffset);
    }
    // TODO doesnt work as intended I believe
    m_segmentsInformation.reserve(c_contiguous_segment_count + 1);
    id_t firstId = partialSegmentTablesCount * c_contiguous_segment_count;
    m_segmentTableOffsets.push_back(m_head);
    for (id_t i = firstId; i < firstId + c_contiguous_segment_count; i++) {
        m_segmentsInformation.push_back({i});
        write(i);
        write(static_cast<link_t>(0x00));
    }
    // Create last entry as a link, which will be overriden by the next call to this method as per
    // above
    write(c_partial_table_link_id);
    write(static_cast<link_t>(0x00));
}

void SharedMemoryPosix::setSegmentTableLink(const id_t id, const link_t position) {
    size_t partialTableIdx = id / c_contiguous_segment_count;
    id_t partialTableStart = m_segmentTableOffsets.at(partialTableIdx);
    id_t writeOffset =
        partialTableStart + (id % c_contiguous_segment_count) * 2 * sizeof(id_t) + sizeof(id_t);
    write(position, writeOffset);
}

void SharedMemoryPosix::updateValidLength(id_t segmentId) {
    const SegmentInformation& segmentInformation = findSegmentInformation(segmentId);
    size_t validLength = segmentInformation.getValidLength();
    position_t validLengthPosition = segmentInformation.getValidLengthPosition();
    // TODO holy fucking ass code
    write(validLength, validLengthPosition);
}

std::vector<std::byte> SharedMemoryPosix::read(position_t position, size_t numBytes) const {
    if ((position + numBytes) > m_size) {
        throw std::out_of_range("End position to read is outside the shm region");
    }
    return {m_memoryStart + position, m_memoryStart + position + numBytes};
}

bool SharedMemoryPosix::isStartValid(void) const {
    std::vector<std::byte> shmStart = read(0, 8);
    return isContainedAt(shmStart, c_UnBARableAI_magic) && isContainedAt(shmStart, c_version, 4);
}

bool SharedMemoryPosix::isSegmentTableValid(position_t start, id_t firstId) const {
    std::vector<std::byte> partialSegmentTable = read(start, c_segment_table_size);
    // An entry consisting of (id) -> (link)
    constexpr size_t c_entry_size = sizeof(id_t) + sizeof(link_t);
    for (id_t i = 0; i < c_contiguous_segment_count; i++) {
        if (!isContainedAt(partialSegmentTable, firstId + i, i * c_entry_size)) {
            // TODO remove
            // std::cout << dataViewToUnsigned<position_t>(
            //    std::span{partialSegmentTable}.subspan(i * c_entry_size));
            return false;
        }
    }
    if (!isContainedAt(partialSegmentTable, c_partial_table_link_id,
                       c_segment_table_size - c_entry_size)) {
        return false;
    }
    position_t continuation = SI<position_t>::deserialize(
        std::span{partialSegmentTable}
            .subspan<c_segment_table_size - sizeof(link_t), sizeof(link_t)>());
    return continuation ? isSegmentTableValid(continuation, firstId + c_contiguous_segment_count)
                        : true;
}

};  // namespace memory
};  // namespace UnBARableAINS
