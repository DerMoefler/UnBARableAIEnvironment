#pragma once

#include <optional>
#include <random>
#include <string_view>
#include <stdexcept>
#include <variant>
#include <vector>
#include <utility>

#include "action.h"
#include "engine_status.h"
#include "grpc_connection.hpp"
#include "serialization/serialize_information.h"
#include "serialization/trivial_serializable.hpp"
#include "unit_data.h"

#include <memory/shared_memory.h>
#include <memory/shared_memory_types.h>
#include <memory/shared_memory_posix.h>

#include <serialization/action.tpp>
#include <serialization/enum.tpp>
#include <serialization/floating_point.tpp>
#include <serialization/grpc_connection.tpp>
#include <serialization/optional.tpp>
#include <serialization/unit_data.tpp>
// The optional include must be after UnitData and

#include <utility/typelist.h>

static_assert(
    UnBARableAINS::serialization::Serializable<std::optional<UnBARableAINS::GrpcConnection>>,
    "Doesnt work");

namespace UnBARableAINS::memory {

/**
 * \brief Provides a BAR-specific interface to the generic shared-memory system.
 *
 * The expected order of UnitData objects in shared memory is:
 *
 * 0. EngineStatus
 * 1. gRpcConnection
 * 2. All units belonging to the own team.
 * 3. All currently visible enemy units.
 * 4. Actions for ally units.
 *
 * The team_id of the first stored UnitData object is interpreted as
 * the own team ID.
 * \todo Could also just write team id to the memory tbh.
 * \todo Could implement a cache for free Unit/Action slots.
 */
class BarSharedMemory {
public:
    /// \brief Alias for the UnitData type.
    using UnitData = UnBARableAINS::unit::UnitData;

    /// \brief Alias for the optional UnitData type used by this shared-memory interface.
    using UnitDataOpt = std::optional<UnitData>;

    /// \brief Alias for the optional Action type used by this shared-memory interface.
    using ActionOpt = std::optional<Action>;

    static_assert(serialization::TrivialSerializable<UnitDataOpt> &&
                      serialization::TrivialSerializable<ActionOpt>,
                  "Both UnitDataOpt and ActionOpt rely on SharedMemory::update which requires its "
                  "types to be TrivialSerializables (at least currently).");

    /**
     * \brief Concrete shared-memory implementation used internally.
     *
     * The implementation uses POSIX shared memory and supports serialization
     * of the named types.
     * \note We use std::optional UnitDatas and Actions rather than the plain types.
     */
    using Impl =
        SharedMemory<SharedMemoryPosix, EngineStatus, GrpcConnection, UnitDataOpt, ActionOpt>;

    /// \brief Type alias for the ValueVariant of the implementation, i.e.
    /// std::variant<SupportedTypes...>
    using ValueVariant = typename Impl::ValueVariant;

    /**
     * \brief Type used to identify serialized objects in shared memory.
     */
    using SerializableId = memory::id_t;

    /// \brief SerializableId for the EngineStatus.
    static constexpr SerializableId c_engine_status_id = 0;

    /// \brief SerializableId for the gRpcConnection.
    static constexpr SerializableId c_grpc_connection_id = 1;

    /**
     * \brief Creates a new POSIX shared-memory region with EngineStatus, GrpcConnection and empty
     * units / actions.
     * \param name Name of the shared-memory region
     * \param status Initial engine status.
     * \param connection GrpcConnection, i.e. configuration like port number.
     * \param initialNumAllyUnits Number of initial ally units.
     * \param initialNumEnemyUnits Number of initial enemy units.
     * \return BarSharedMemory object connected to the newly created region
     *
     * This constructor writes the \p status and \p connection into their respective Serializables
     * (see c_engine_status_id and c_grpc_connection_id). It also creates "slots", i.e.
     * std::optionals, for UnitDatas as well as Actions. These are initialized to std::nullopt to
     * begin with, but the entire memory is allocated for those.
     *
     * The name must not already refer to an existing POSIX shared-memory
     * region. POSIX shared-memory names should start with a forward slash,
     * for example "/bar_shared_memory".
     *
     * \throws std::system_error if the shared-memory region cannot be created, std::runtime_error
     * if assigned ids for \p status or \p connection do not match \ref c_engine_status_id or \ref
     * c_grpc_connection_id respectively.
     */
    static BarSharedMemory create(const std::string_view name, const EngineStatus status,
                                  const GrpcConnection& connection,
                                  const std::size_t initialNumAllyUnits,
                                  const std::size_t initialNumEnemyUnits);

    /**
     * \brief Opens an existing POSIX shared-memory region.
     * \param name name of the existing shared-memory region
     * \return BarSharedMemory object connected to the opened region
     *
     * The shared-memory region must already have been created by another
     * process or by an earlier call to create().
     *
     * \throws std::system_error if the shared-memory region cannot be opened, std::runtime_error if
     * there is not exactly one instance of EngineStatus/GrpcConnection or the ids do not match \ref
     * c_engine_status_id or \ref c_grpc_connection_id respectively.
     */
    static BarSharedMemory open(const std::string_view name);

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
     * \ref Update the information about the shared-memory region.
     * \returns Whether or not any changes have been detected.
     * \note Value changes are not reported, only changes in size.
     */
    inline bool update(void) { return m_sharedMemory.update(); }

    /**
     * \brief Updates (i.e. overrides) an Action object already existing in the shared-memory.
     * \param actionId Id of the action to update.
     * \param action ActionOpt that should be serialized and stored.
     * \throws std::out_of_range if the \p actionId is out of range.
     * \throws std::runtime_error if the Serializable at \p actionId is not actually an
     * ActionOpt.
     */
    void updateAction(const SerializableId actionId, const ActionOpt& action);

    /**
     * \brief Updates (i.e. overrides) a UnitData object already existing in the shared-memory.
     * \param unitId Id of the unit to update.
     * \param unitData UnitDataOpt that should be serialized and stored
     * \throws std::out_of_range if the \p unitId is out of range.
     * \throws std::runtime_error if the Serializable at \p unitId is not actually a UnitDataOpt.
     *
     * The team_id of the first UnitData object written through this wrapper
     * is stored as the own team ID.
     */
    void updateUnitData(const SerializableId unitId, const UnitDataOpt& unitData);

    /**
     * \brief Sets all unit and action opts to std::nullopt.
     * \todo Test
     */
    void clearUnitsAndActions(void);

    /**
     * \brief Get a SerializableId for a UnitDataOpt that is empty, i.e. contains std::nullopt.
     * \param excludeIds A set of Ids that should be excluded from the search.
     * \returns A SerializableId or std::nullopt if all UnitDataOpt Serializables contain data.
     *
     * The \p exlucdeIds param is useful if you already assigned an empty unit slot to some unit but
     * have not actually yet written it.
     */
    std::optional<SerializableId> getEmptyUnitSlot(
        const std::set<SerializableId>& excludeIds) const;

    /**
     * \brief Update the engine status in the shared-memory.
     * \param status Current status of the game engine
     */
    void updateEngineStatus(const EngineStatus status);

    /// \brief Getter for EngineStatus.
    EngineStatus getEngineStatus(void) const;

    /// \brief Getter for GrpcConnection.
    GrpcConnection getGrpcConnection(void) const;

    /**
     * \brief Reads all stored values of the requested type.
     * \tparam T supported serializable value type
     * \param[out] ids Optional output parameter containing the ids of the read Serializables. Must
     * be empty on call.
     * \return values of type T in serializable-ID order
     *
     * Entries containing other supported types are skipped.
     *
     * The current implementation assumes that serializable IDs begin
     * at zero, are contiguous and cannot be deleted.
     * \todo Has to be refactored if deletion was to be used.
     */
    template <typename T>
        requires(Typelist::Contains_MF<typename Impl::SupportedTypelist, T>::value)
    std::vector<T> readAll(std::vector<SerializableId>* ids = nullptr) const {
        if (ids) {
            assert(ids->empty() && "Input parameter ids must be empty");
        }

        std::vector<T> values;

        SerializableId serializableId = 0;

        while (true) {
            try {
                const auto& layoutVariant = m_sharedMemory.getLayout(serializableId);

                const bool containsRequestedType =
                    std::holds_alternative<serialization::Layout<T>>(layoutVariant);

                if (containsRequestedType) {
                    auto valueVariant = m_sharedMemory.read(serializableId);

                    /*
                     * Das Layout und der deserialisierte Wert müssen
                     * denselben Typ besitzen.
                     */
                    if (!std::holds_alternative<T>(valueVariant)) {
                        throw std::runtime_error(
                            "Layout type and deserialized value "
                            "type do not match.");
                    }

                    values.push_back(std::get<T>(std::move(valueVariant)));
                    if (ids) {
                        ids->push_back(serializableId);
                    }
                }
            } catch (const std::out_of_range&) {
                /*
                 * getLayout() wirft std::out_of_range, sobald die
                 * Serializable-ID außerhalb von m_layouts liegt.
                 */
                break;
            }

            ++serializableId;
        }

        return values;
    }

    template <serialization::Serializable S>
        requires(Typelist::Contains_MF<typename Impl::SupportedTypelist, S>::value)
    std::vector<SerializableId> getIdsForType(std::type_identity<S> typeId) const {
        const auto serializablesCount = m_sharedMemory.serializablesCount();

        std::vector<SerializableId> ids;
        ids.reserve(serializablesCount);
        for (std::size_t i = 0; i < serializablesCount; i++) {
            if (m_sharedMemory.containsType(typeId, i)) {
                ids.push_back(i);
            }
        }

        ids.shrink_to_fit();
        return ids;
    }

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

    inline const std::string& getName(void) const { return m_sharedMemory.getName(); }

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
     * \brief Helper method to throw (and log an error) for invalid ids.
     * \param expectedId The id that was expected for the Serializable.
     * \param actualid The id that Serializable was actually assigned.
     * \param name Name of the Serializable to put into the error message, e.g "EngineStatus".
     * \throws std::runtime_error if expetectedId != actualId.
     */
    static void throwOnInvalidId(const SerializableId expectedId, const SerializableId actualId,
                                 const std::string_view name);

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
