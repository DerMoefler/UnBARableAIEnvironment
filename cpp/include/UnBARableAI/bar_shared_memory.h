#pragma once

#include <optional>
#include <string_view>
#include <stdexcept>
#include <variant>
#include <vector>
#include <utility>

#include "action.h"
#include "engine_status.h"
#include "grpc_connection.hpp"
#include "serialization/serialize_information.h"
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
 */
class BarSharedMemory {
public:
    using UnitData = UnBARableAINS::unit::UnitData;

    /// \brief Alias for the UnitData type used by this shared-memory interface.
    using UnitDataOpt = std::optional<UnitData>;

    /// \brief Alias for the Action type used by this shared-memory interface.
    using ActionOpt = std::optional<Action>;

    /**
     * \brief Concrete shared-memory implementation used internally.
     *
     * The implementation uses POSIX shared memory and supports serialization
     * of UnitData and Action objects.
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
     * \throws std::system_error if the shared-memory region cannot be created
     */
    static BarSharedMemory create(std::string_view name, const EngineStatus status,
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
    SerializableId writeAction(const ActionOpt& action);

    /**
     * \brief Writes a UnitData object to shared memory.
     * \param unitData unit data that should be serialized and stored
     * \return serializable ID assigned to the stored UnitData object
     *
     * The team_id of the first UnitData object written through this wrapper
     * is stored as the own team ID.
     *
     */
    SerializableId writeUnitData(const UnitDataOpt& unitData);

    /**
     * \brief Writes an engine status to shared memory.
     * \param status current status of the game engine
     * \return serializable ID assigned to the stored engine status
     */
    SerializableId writeEngineStatus(EngineStatus status);

    /**
     * \brief Reads all stored values of the requested type.
     * \tparam T supported serializable value type
     * \return values of type T in serializable-ID order
     *
     * Entries containing other supported types are skipped.
     *
     * The current implementation assumes that serializable IDs begin
     * at zero, are contiguous and cannot be deleted.
     */
    template <typename T>
        requires(Typelist::Contains_MF<typename Impl::SupportedTypelist, T>::value)
    std::vector<T> readAll() {
        std::vector<T> values;

        SerializableId serializableId = 0;

        while (true) {
            try {
                auto& layoutVariant = m_sharedMemory.getLayout(serializableId);

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
