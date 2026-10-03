#include "id_allocator.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>

namespace id {

id_t IdAllocator::allocate(void) {
    id_t id = 0;
    if (!m_freedIds.empty()) {
        auto it = m_freedIds.begin();
        id = *it;
        m_freedIds.erase(it);
    }
    else {
        id = m_nextId++;
    }
    m_usedIds.insert(id);
    return id;
}

bool IdAllocator::release(id_t id) {
    auto erased = m_usedIds.erase(id);
    if (erased > 0) {
        m_freedIds.insert(id);
        return true;
    }
    else {
        return false;
    }
}

void IdAllocator::setUsedIds(const std::set<id_t>& usedIds) {
    std::cout << "IdAllocator::setUsedIds: Begin\n";
    if (!isEmpty()) {
        throw std::logic_error("Set must be empty to use 'setUsedIds' method.");
    }
    m_usedIds = usedIds;
    if (m_usedIds.empty()) {
        return;
    }

    m_nextId = *usedIds.rbegin() + 1;

    id_t expected{0};
    for (id_t id : usedIds) {
        while (expected < id) {
            m_freedIds.insert(expected);
            expected++;
        }
        expected++;
    }
    std::cout << "IdAllocator::setUsedIds: End\n";
}

id_t IdAllocator::idAt(id_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Index out of bounds.");
    }

    id_t id = index;

    for (id_t freeId : m_freedIds) {
        if (freeId > id) break;
        id++;
    }

    return id;
}

std::optional<id_t> IdAllocator::indexAt(id_t id) const {
    if (id >= m_nextId) {
        throw std::out_of_range("Id out of bounds.");
    }
    if (m_freedIds.contains(id)) {
        return std::nullopt;
    }

    id_t freedBefore = 0;

    for (id_t freeId : m_freedIds) {
        if (freeId >= id) break;
        freedBefore++;
    }
    return id - freedBefore;
}

}  // namespace id
