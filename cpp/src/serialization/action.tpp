#ifndef ACTION_TPP_
#define ACTION_TPP_

#include <cstdint>
#include <type_traits>

#include "serialization/serialize_information.h"
#include "UnBARableAI/action.h"

namespace UnBARableAINS {

namespace serialization {

template <>
struct SerializeInformation<Action> {
    using Type = Action;

    inline static constexpr bool c_serializable = true;
    inline static constexpr bool c_inline = true;

    struct F_UnitId : detail::Field_t {
        using Type = uint32_t;
    };

    struct F_TeamId : detail::Field_t {
        using Type = uint32_t;
    };

    struct F_AllyTeamId : detail::Field_t {
        using Type = uint32_t;
    };

    struct F_TargetUnitId : detail::Field_t {
        using Type = uint32_t;
    };

    using Fields = detail::Fields<F_UnitId, F_TeamId, F_AllyTeamId, F_TargetUnitId>;

    inline static constexpr uint32_t get(std::type_identity<F_UnitId>,
                                         const Type& action) noexcept {
        return action.unit_id;
    }

    inline static constexpr uint32_t get(std::type_identity<F_TeamId>,
                                         const Type& action) noexcept {
        return action.team_id;
    }

    inline static constexpr uint32_t get(std::type_identity<F_AllyTeamId>,
                                         const Type& action) noexcept {
        return action.ally_team_id;
    }

    inline static constexpr uint32_t get(std::type_identity<F_TargetUnitId>,
                                         const Type& action) noexcept {
        return action.target_unit_id;
    }

    inline static constexpr Type constructFromFields(uint32_t unit_id, uint32_t team_id,
                                                     uint32_t ally_team_id,
                                                     uint32_t target_unit_id) noexcept {
        // TODO Fix action id
        return Action{unit_id, team_id, ally_team_id, ActionId::Attack, target_unit_id};
    }
};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
