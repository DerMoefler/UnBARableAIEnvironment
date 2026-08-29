#ifndef ID_ALLOCATOR_H_
#define ID_ALLOCATOR_H_

#include <optional>
#include <set>

#include "id_types.hpp"

namespace id {

/**
 * \brief A simple class for id allocations, giving out continous id's and reusing freed ones.
 */
class IdAllocator {
public:
    /**
     * \brief Get next id and allocate it in the buffer.
     */
    id_t allocate(void);

    /**
     * \brief Release a specific id (must not actually be allocated).
     * \param id An id to be released.
     * \returns Whether the id was released.
     */
    bool release(id_t id);

    /// \brief Get number of allocated ids.
    inline size_t size(void) const { return m_nextId - m_freedIds.size(); }

    /**
     * \brief Get the id corresponding to some index.
     *
     * If the ids {0, 2, 4, 5} were allocated, idAt(2) would be 4.
     */
    id_t idAt(id_t index) const;

    /**
     * \brief Get the (optional) index corresponding to some id.
     *
     * If the ids {0, 2, 4, 5} were allocated, indexAt(2) would be 1 and indexAt(3) would be
     * std::nullopt.
     */
    std::optional<id_t> indexAt(id_t id) const;

private:
    id_t m_nextId = 0;
    std::set<id_t> m_usedIds;
    std::set<id_t> m_freedIds;
};

}  // namespace id

#endif  // ID_ALLOCATOR_H_
