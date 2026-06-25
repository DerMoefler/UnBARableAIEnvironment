#pragma once
// #include <asm-generic/errno.h>
#include <string_view>

#include "shared_memory_impl.h"
#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace memory {

template<SharedMemoryImpl T>
class SharedMemory {
public:
    /**
     * \brief Creates a shared memory region.
     */
    SharedMemory(std::string_view name)
        : m_sharedMemoryImpl(name)
    {}

    template<serialization::Serializable S>
    void write(const S& value) {
        auto layout = serialization::Layout<S>{value};
        createSegments(layout);
    }

private:
    template<serialization::Serializable S>
    void createSegments(const serialization::Layout<S>& layout) {
        decltype(auto) nodes = layout.getNodes();
        size_t inlinedSize;
        std::apply([&](const auto&... node){
            auto forEachNode = [&](const auto& node) {
                using Tag = typename std::remove_cvref_t<decltype(node)>::Tag;
                // --- Recursion ---
                if constexpr (serialization::detail::MultiField<Tag>) {
                    const auto& children = node.children;
                    for (const auto& child : children) {
                        if (child) {
                            createSegments(*child);
                        }
                    }
                }
                else {
                    if (node.child) {
                        createSegments(*node.child);
                    }
                }
                // --- end recursion ---
                inlinedSize = node.inlineSize;
                m_sharedMemoryImpl.createSegment(inlinedSize);
            };

            (forEachNode(node), ...);
            
        }, nodes);
    }

    T m_sharedMemoryImpl;

};

}; // namespace memory

}; // namespace UnBARableAI
