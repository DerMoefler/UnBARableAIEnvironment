#pragma once
#include <cstdint>
#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "memory/shared_memory_types.h"
#include "serialization/serialize_information.h"
#include "shared_memory_impl.h"

namespace UnBARableAINS {

namespace memory {

template <SharedMemoryImpl T, typename... SupportedTypes>
class SharedMemory {
private:
    using LayoutVariant = std::variant<serialization::Layout<SupportedTypes>...>;

public:
    /**
     * \brief Creates a shared memory region.
     */
    SharedMemory(std::string_view name) : m_sharedMemoryImpl(name) {}

    template <serialization::Serializable S>
    void write(const S& value) {
        auto layout = std::make_shared<serialization::Layout<S>>(value);

        createSegments(layout);

        m_layouts.push_back(*layout);
    }

private:
    template <serialization::Serializable S>
    void createSegments(std::shared_ptr<serialization::Layout<S>> layout) {
        memory::id_t segmentId = m_sharedMemoryImpl.createSegment(layout->getInlinedSize());
        layout->setSegmentId(segmentId);

        auto funcBase = [segmentId](auto& child) { child->setSegmentId(segmentId); };
        auto funcRecursive = [&](auto& child) { createSegments(child); };

        layout->applyToNodes(funcBase, funcRecursive);
    }

    std::vector<LayoutVariant> m_layouts;

    T m_sharedMemoryImpl;
};

};  // namespace memory

};  // namespace UnBARableAINS
