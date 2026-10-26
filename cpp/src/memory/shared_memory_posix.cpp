#include "shared_memory_posix.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
#include "memory/segment_header.hpp"
#include "memory/shm_header.hpp"
#include "serialization/serialize_information.h"
#include "serialization/trivial_serializable.hpp"
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
    SharedMemoryPosix result(name);
    result.m_id = shm_open(result.m_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
    if (result.m_id == -1) {
        throw std::system_error(errno, std::generic_category(), "shm_open failed");
    }
    // Initially allocate 4096 bytes
    result.increaseSize();

    result.write(ShmHeader{}, 0);

    result.extendSegmentTable();

    SPDLOG_LOGGER_INFO(logger, "Created shared memory \"{}\"", name);
    return result;
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
    , m_memoryStart(std::exchange(other.m_memoryStart, nullptr))
    , m_size(std::exchange(other.m_size, 0))
    , m_segmentTable(std::move(other.m_segmentTable)) {}

SharedMemoryPosix& SharedMemoryPosix::operator=(SharedMemoryPosix&& other) {
    if (this == &other) {
        return *this;
    }
    release();

    m_name = std::move(other.m_name);
    m_id = std::exchange(other.m_id, -1);
    m_memoryStart = std::exchange(other.m_memoryStart, nullptr);
    m_size = std::exchange(other.m_size, 0);
    m_segmentTable = std::move(other.m_segmentTable);

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
    // TODO implement
    // if (!isSegmentTableValid())
    //     throw std::runtime_error("Shared memory's segment table is invalid.");

    // const auto oldSegmentsInformation = m_segmentsInformation;
    // if (!initializeSegmentsInformation())
    //     throw std::runtime_error("Initializing segments information failed");

    // if (oldSegmentsInformation != m_segmentsInformation) {
    //     change = true;
    //     initializeIdAllocator();
    // }

    SPDLOG_LOGGER_DEBUG(logger, "Updated shared memory \"{}\"", m_name);
    return change;
}

id_t SharedMemoryPosix::createSegment(const std::size_t size) {
    SPDLOG_LOGGER_TRACE(logger, "Creation of segment with size {} request.", size);
    if (!size) {
        constexpr std::string_view message = "Cannot create empty segment";
        SPDLOG_LOGGER_ERROR(logger, message);
        throw std::length_error(std::string(message));
    }

    if (m_segmentTable.isFull()) {
        extendSegmentTable();
    }

    assert(!m_segmentTable.isFull() && "Segment table extension did not work properly.");

    const SegmentInformation& newSegment = allocateNewSegment(size);

    write(newSegment.getHeader(), newSegment.getMemoryStart());

    SPDLOG_LOGGER_DEBUG(logger, "New segment with id={} created.", newSegment.getId());
    return newSegment.getId();
}

void SharedMemoryPosix::appendToSegment(const id_t segmentId, DataView data) {
    // std::cout << "SharedMemoryPosix::apendToSegment" << std::endl;
    const SegmentInformation& segment = m_segmentTable.getSegmentInformation(segmentId);
    for (size_t i = 0; i < data.size(); i++) {
        write(static_cast<uint8_t>(data[i]), segment.headPosition());
        m_segmentTable.advanceSegmentInformationHead(segmentId, 1);
    }
    updateValidLength(segmentId);
}

std::vector<std::byte> SharedMemoryPosix::readSegment(const id_t segmentId) const {
    const SegmentInformation& segmentInformation = m_segmentTable.getSegmentInformation(segmentId);

    const position_t dataStart = segmentInformation.dataStart();
    const std::size_t dataSize = segmentInformation.validLength();

    if (!dataSize) {
        return std::vector<std::byte>{};
    }

    std::vector<std::byte> rawSegmentData(dataSize);

    std::memcpy(rawSegmentData.data(), m_memoryStart + dataStart, dataSize);

    return rawSegmentData;
}

id_t SharedMemoryPosix::writeSegment(DataView data) {
    id_t segmentId = createSegment(data.size());
    appendToSegment(segmentId, data);
    return segmentId;
}

void SharedMemoryPosix::deleteSegment(id_t segmentId) {
    // TODO implement
}

std::optional<id_t> SharedMemoryPosix::getLinkedSegment(id_t segmentId, position_t offset) const {
    const SegmentInformation& segmentInformation = m_segmentTable.getSegmentInformation(segmentId);
    // Link cannot be valid if data hasnt even been written
    if (offset + sizeof(link_t) > segmentInformation.validLength()) {
        return std::nullopt;
    }
    // TODO readSegmentAt function or something
    position_t dataPosition = segmentInformation.dataStart() + offset;
    auto serialized = read(dataPosition, sizeof(link_t));
    assert(serialized.size() == sizeof(link_t) && "Link's size must be sizeof(link_t) bytes");
    auto linkedSegmentId = SI<link_t>::deserialize(
        std::span<const std::byte, sizeof(link_t)>{serialized.data(), sizeof(link_t)});
    const SegmentInformation& linkedSegmentInformation =
        m_segmentTable.getSegmentInformation(linkedSegmentId);
    if (linkedSegmentInformation.isValid()) {
        return linkedSegmentId;
    }
    else {
        return std::nullopt;
    }
}

SharedMemoryPosix::SharedMemoryPosix(std::string_view name)
    : m_name(name) {}

void SharedMemoryPosix::increaseSize(const size_t size) {
    ftruncate(m_id, m_size + size);
    m_size += size;
    map();
}

void SharedMemoryPosix::increaseSizeIfNecessary(void) {
    position_t head = m_segmentTable.getHead();
    if (head > m_size) {
        const std::size_t requiredSize = head - static_cast<position_t>(m_size);
        const std::size_t sizeIncrease = (requiredSize / c_size_increase) * (c_size_increase + 1);
        increaseSize(sizeIncrease);
        SPDLOG_LOGGER_TRACE(logger, "Increased size by {}", sizeIncrease);
    }
}

const SegmentInformation& SharedMemoryPosix::allocateNewSegment(const std::size_t dataSize) {
    const SegmentInformation& newSegment = m_segmentTable.allocate(dataSize);
    increaseSizeIfNecessary();
    updateSegmentTableLink(newSegment.getId(), static_cast<link_t>(newSegment.getMemoryStart()));
    return newSegment;
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

    m_name = {};
    m_size = 0;
    m_segmentTable = SegmentTable();
}

void SharedMemoryPosix::extendSegmentTable(void) {
    auto newPartialTable = m_segmentTable.extend();
    write(newPartialTable, newPartialTable.getOffset());
    increaseSizeIfNecessary();
    SPDLOG_LOGGER_TRACE(logger, "Extended SegmentTable.");
}

void SharedMemoryPosix::updateSegmentTableLink(const id_t segmentId,
                                               const std::optional<link_t> link) {
    using F_Link = typename serialization::SerializeInformation<SegmentTableEntry>::F_Link;
    constexpr std::size_t linkOffset = serialization::getFieldOffset(
        std::type_identity<SegmentTableEntry>{}, std::type_identity<F_Link>{});

    const position_t entryPosition = m_segmentTable.getSegmentEntryPosition(segmentId);
    const position_t entryLinkPosition = entryPosition + static_cast<position_t>(linkOffset);

    write(link, entryLinkPosition);
}

void SharedMemoryPosix::updateValidLength(const id_t segmentId) {
    using F_Head = typename serialization::SerializeInformation<SegmentHeader>::F_Head;
    static_assert(std::same_as<typename F_Head::Type, position_t>, "Implementation changed");
    constexpr std::size_t headFieldOffset = serialization::getFieldOffset(
        std::type_identity<SegmentHeader>{}, std::type_identity<F_Head>{});

    const SegmentInformation& segmentInformation = m_segmentTable.getSegmentInformation(segmentId);

    const std::size_t segmentHead = segmentInformation.validLength();
    position_t segmentHeadPosition = segmentInformation.getMemoryStart() + headFieldOffset;
    write(static_cast<position_t>(segmentHead), segmentHeadPosition);
}

std::vector<std::byte> SharedMemoryPosix::read(position_t position, size_t numBytes) const {
    if ((position + numBytes) > m_size) {
        throw std::out_of_range("End position to read is outside the shm region");
    }
    return {m_memoryStart + position, m_memoryStart + position + numBytes};
}

bool SharedMemoryPosix::isStartValid(void) const {
    // TODO refactor to use deserialize maybe
    constexpr std::size_t shmHeaderSize =
        serialization::getSerializableSize(std::type_identity<ShmHeader>{});
    std::vector<std::byte> serializedHeader = read(0, shmHeaderSize);
    try {
        serialization::deserialize<ShmHeader>(
            std::span<const std::byte, shmHeaderSize>{serializedHeader.data(), shmHeaderSize});
        return true;
    } catch (const std::invalid_argument& exception) {
        return false;
    }
}

};  // namespace memory

};  // namespace UnBARableAINS
