#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "action.h"
#include "unit_data.h"

#include <memory/shared_memory.h>
#include <memory/shared_memory_types.h>
#include <memory/shared_memory_posix.h>

#include <serialization/floating_point.tpp>
#include <serialization/unit_data.tpp>
#include <serialization/action.tpp>

namespace UnBARableAINS::memory {

/**
 * \brief Provides a BAR-specific interface to the generic shared-memory system.
 *
 * The expected order of UnitData objects in shared memory is:
 *
 * 1. All units belonging to the own team.
 * 2. All currently visible enemy units.
 *
 * The team_id of the first stored UnitData object is interpreted as
 * the own team ID.
 */
class BarSharedMemory {
public:
    /**
     * \brief Alias for the UnitData type used by this shared-memory interface.
     */
    using UnitData = UnBARableAINS::unit::UnitData;

    /**
     * \brief Alias for the Action type used by this shared-memory interface.
     */
    using Action = UnBARableAINS::Action;

    /**
     * \brief Concrete shared-memory implementation used internally.
     *
     * The implementation uses POSIX shared memory and supports serialization
     * of UnitData and Action objects.
     */
    using Impl = SharedMemory<SharedMemoryPosix, UnitData, Action>;

    /// \brief Type alias for the ValueVariant of the implementation, i.e.
    /// std::variant<SupportedTypes...>
    using ValueVariant = typename Impl::ValueVariant;

    /**
     * \brief Type used to identify serialized objects in shared memory.
     */
    using SerializableId = memory::id_t;

    /**
     * \brief Creates a new POSIX shared-memory region.
     * \param name name of the shared-memory region
     * \return BarSharedMemory object connected to the newly created region
     *
     * The name must not already refer to an existing POSIX shared-memory
     * region. POSIX shared-memory names should start with a forward slash,
     * for example "/bar_shared_memory".
     *
     * \throws std::system_error if the shared-memory region cannot be created
     */
    static BarSharedMemory create(std::string_view name);

    /**
     * \brief Opens an existing POSIX shared-memory region.
     * \param name name of the existing shared-memory region
     * \return BarSharedMemory object connected to the opened region
     *
     * The shared-memory region must already have been created by another
     * process or by an earlier call to create().
     *
     * \throws std::system_error if the shared-memory region cannot be opened
     */
    static BarSharedMemory open(std::string_view name);

    /**
     * \brief Removes the name of a POSIX shared-memory region.
     * \param name name of the shared-memory region to remove
     *
     * Existing mappings may remain valid until the connected processes
     * release them.
     *
     * \throws std::system_error if the shared-memory region cannot be removed
     */
    static void remove(std::string_view name);

    /**
     * \brief Deleted copy constructor.
     *
     * A BarSharedMemory object owns a shared-memory connection and therefore
     * cannot be copied.
     */
    BarSharedMemory(const BarSharedMemory&) = delete;

    /**
     * \brief Deleted copy-assignment operator.
     * \return reference to this object
     *
     * A BarSharedMemory object owns a shared-memory connection and therefore
     * cannot be copied.
     */
    BarSharedMemory& operator=(const BarSharedMemory&) = delete;

    /**
     * \brief Move constructor.
     * \param other BarSharedMemory object whose resources are transferred
     *
     * After the move, other remains valid but no longer owns the transferred
     * shared-memory resources.
     */
    BarSharedMemory(BarSharedMemory&& other) noexcept = default;

    /**
     * \brief Move-assignment operator.
     * \param other BarSharedMemory object whose resources are transferred
     * \return reference to this object
     */
    BarSharedMemory& operator=(BarSharedMemory&& other) noexcept = default;

    /**
     * \brief Destroys the BarSharedMemory wrapper.
     *
     * The underlying shared-memory implementation releases its local
     * resources. Removing the POSIX shared-memory name should be performed
     * explicitly through remove().
     */
    ~BarSharedMemory() = default;

    /**
     * \brief Writes an Action object to shared memory.
     * \param action action that should be serialized and stored
     * \return serializable ID assigned to the stored Action object
     *
     * The returned serializable ID identifies the object inside the generic
     * shared-memory system.
     */
    SerializableId writeAction(const Action& action);

    /**
     * \brief Writes a UnitData object to shared memory.
     * \param unitData unit data that should be serialized and stored
     * \return serializable ID assigned to the stored UnitData object
     *
     * The team_id of the first UnitData object written through this wrapper
     * is stored as the own team ID.
     *
     */
    SerializableId writeUnitData(const UnitData& unitData);

    /**
     * \brief Reads all Action objects in serializable-ID order.
     * \return vector containing all Action objects stored in shared memory
     *
     * UnitData entries are skipped. The method currently assumes that
     * serializable IDs start at zero, are contiguous and are not deleted.
     *
     * \todo Implement SharedMemory::read<Action>(SerializableId).
     */
    std::vector<Action> readAllActions();

    /**
     * \brief Reads all UnitData objects in serializable-ID order.
     * \return vector containing all UnitData objects stored in shared memory
     *
     * Action entries are skipped. The returned vector preserves the order in
     * which the UnitData objects appear in the shared-memory layout table.
     *
     * The method currently assumes that serializable IDs start at zero, are
     * contiguous and are not deleted.
     *
     * \todo Implement SharedMemory::read<UnitData>(SerializableId).
     */
    std::vector<UnitData> readAllUnits();

    /**
     * \brief Returns the team ID of the first stored UnitData object.
     * \return team_id of the first UnitData object
     *
     * If the team ID was not set by writeUnitData(), the first UnitData
     * object is read from shared memory to reconstruct the own team ID.
     *
     * \throws std::runtime_error if shared memory contains no UnitData object
     */
    int getOwnTeamId();

private:
    /**
     * \brief Constructs the BAR-specific wrapper from a shared-memory implementation.
     * \param impl initialized generic shared-memory implementation
     *
     * This constructor is private because BarSharedMemory objects should be
     * constructed through create() or open().
     */
    explicit BarSharedMemory(Impl impl);

    /**
     * \brief Resolves and caches the own team ID.
     * \return team_id of the first stored UnitData object
     *
     * If the team ID is already cached, the cached value is returned.
     * Otherwise, the first UnitData object is read from shared memory and its
     * team_id is stored in m_ownTeamId.
     *
     * \throws std::runtime_error if shared memory contains no UnitData object
     */
    int resolveOwnTeamId();

    /**
     * \brief Generic shared-memory implementation used by this wrapper.
     */
    Impl m_sharedMemory;

    /**
     * \brief Cached team ID of the first stored UnitData object.
     */
    std::optional<int> m_ownTeamId;
};

}  // namespace UnBARableAINS::memory
