#ifndef UNIT_DATA_TPP_
#define UNIT_DATA_TPP_

#include <cstdint>
#include <type_traits>

#include "serialization/serialize_information.h"
#include "UnBARableAI/unit_data.h"

namespace UnBARableAINS {

namespace serialization {

template <>
struct SerializeInformation<unit::UnitData> {
    using Type = unit::UnitData;

    inline static constexpr bool c_serializable = true;
    inline static constexpr bool c_inline = true;

    struct F_UnitId : public detail::Field_t {
        using Type = uint32_t;
    };

    struct F_UnitDefId : public detail::Field_t {
        using Type = uint32_t;
    };

    struct F_TeamId : public detail::Field_t {
        using Type = uint32_t;
    };

    struct F_AllyTeamId : public detail::Field_t {
        using Type = uint32_t;
    };

    struct F_Health : public detail::Field_t {
        using Type = float;
    };

    struct F_MaxHealth : public detail::Field_t {
        using Type = float;
    };

    struct F_PosX : public detail::Field_t {
        using Type = float;
    };

    struct F_PosY : public detail::Field_t {
        using Type = float;
    };

    struct F_PosZ : public detail::Field_t {
        using Type = float;
    };

    struct F_LosRadius : public detail::Field_t {
        using Type = float;
    };

    struct F_AirLosRadius : public detail::Field_t {
        using Type = float;
    };

    struct F_IsDead : public detail::Field_t {
        using Type = bool;
    };

    struct F_BeingBuilt : public detail::Field_t {
        using Type = bool;
    };

    struct F_BuildProgress : public detail::Field_t {
        using Type = float;
    };

    struct F_CaptureProgress : public detail::Field_t {
        using Type = float;
    };

    struct F_ParalyzeDamage : public detail::Field_t {
        using Type = float;
    };

    using Fields =
        detail::Fields<F_UnitId, F_UnitDefId, F_TeamId, F_AllyTeamId, F_Health, F_MaxHealth, F_PosX,
                       F_PosY, F_PosZ, F_LosRadius, F_AirLosRadius, F_IsDead, F_BeingBuilt,
                       F_BuildProgress, F_CaptureProgress, F_ParalyzeDamage>;

    inline static constexpr uint32_t get(std::type_identity<F_UnitId>,
                                         const Type& unitData) noexcept {
        return unitData.unit_id;
    }

    inline static constexpr uint32_t get(std::type_identity<F_UnitDefId>,
                                         const Type& unitData) noexcept {
        return unitData.unit_def_id;
    }

    inline static constexpr uint32_t get(std::type_identity<F_TeamId>,
                                         const Type& unitData) noexcept {
        return unitData.team_id;
    }

    inline static constexpr uint32_t get(std::type_identity<F_AllyTeamId>,
                                         const Type& unitData) noexcept {
        return unitData.ally_team_id;
    }

    inline static constexpr float get(std::type_identity<F_Health>, const Type& unitData) noexcept {
        return unitData.health;
    }

    inline static constexpr float get(std::type_identity<F_MaxHealth>,
                                      const Type& unitData) noexcept {
        return unitData.max_health;
    }

    inline static constexpr float get(std::type_identity<F_PosX>, const Type& unitData) noexcept {
        return unitData.pos_x;
    }

    inline static constexpr float get(std::type_identity<F_PosY>, const Type& unitData) noexcept {
        return unitData.pos_y;
    }

    inline static constexpr float get(std::type_identity<F_PosZ>, const Type& unitData) noexcept {
        return unitData.pos_z;
    }

    inline static constexpr float get(std::type_identity<F_LosRadius>,
                                      const Type& unitData) noexcept {
        return unitData.los_radius;
    }

    inline static constexpr float get(std::type_identity<F_AirLosRadius>,
                                      const Type& unitData) noexcept {
        return unitData.air_los_radius;
    }

    inline static constexpr bool get(std::type_identity<F_IsDead>, const Type& unitData) noexcept {
        return unitData.is_dead;
    }

    inline static constexpr bool get(std::type_identity<F_BeingBuilt>,
                                     const Type& unitData) noexcept {
        return unitData.being_built;
    }

    inline static constexpr float get(std::type_identity<F_BuildProgress>,
                                      const Type& unitData) noexcept {
        return unitData.build_progress;
    }

    inline static constexpr float get(std::type_identity<F_CaptureProgress>,
                                      const Type& unitData) noexcept {
        return unitData.capture_progress;
    }

    inline static constexpr float get(std::type_identity<F_ParalyzeDamage>,
                                      const Type& unitData) noexcept {
        return unitData.paralyze_damage;
    }

    inline static constexpr Type constructFromFields(
        uint32_t unit_id, uint32_t unit_def_id, uint32_t team_id, uint32_t ally_team_id,
        float health, float max_health, float pos_x, float pos_y, float pos_z, float los_radius,
        float air_los_radius, bool is_dead, bool being_built, float build_progress,
        float capture_progress, float paralyze_damage) {
        return unit::UnitData{
            unit_id,        unit_def_id, team_id,     ally_team_id,   health,
            max_health,     pos_x,       pos_y,       pos_z,          los_radius,
            air_los_radius, is_dead,     being_built, build_progress, capture_progress,
            paralyze_damage};
    }
};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
