from collections.abc import Mapping
from typing import Any

import numpy as np


# One self record, two ally slots, and three enemy slots for a 3v3 match.
SELF_FEATURE_DIM = 8
UNIT_SLOT_FEATURE_DIM = 9
MAX_ALLY_SLOTS = 2
MAX_ENEMY_SLOTS = 3
POSITION_SCALE = 1000.0
OBS_DIM = SELF_FEATURE_DIM + (
    MAX_ALLY_SLOTS + MAX_ENEMY_SLOTS
) * UNIT_SLOT_FEATURE_DIM


def _unit_record(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _field(unit: Any, name: str) -> Any:
    if isinstance(unit, Mapping):
        return unit[name]
    return getattr(unit, name)


def _health_fraction(unit: Any) -> float:
    max_health = max(0.0, float(_field(unit, "max_health")))
    if max_health == 0.0:
        return 0.0
    return float(np.clip(float(_field(unit, "health")) / max_health, 0.0, 1.0))


def _self_features(unit: Any) -> list[float]:
    return [
        float(_field(unit, "unit_def_id")),  # Unit type
        _health_fraction(unit),  # Current health / maximum health
        float(_field(unit, "pos_x")) / POSITION_SCALE,  # Scaled world X
        float(_field(unit, "pos_y")) / POSITION_SCALE,  # Scaled world Y
        float(_field(unit, "pos_z")) / POSITION_SCALE,  # Scaled world Z
        float(_field(unit, "los_radius")) / POSITION_SCALE,  # Scaled sight radius
        float(bool(_field(unit, "is_dead"))),  # Dead-state flag
        float(bool(_field(unit, "being_built"))),  # Construction-state flag
    ]


def _distance_squared(first: Any, second: Any) -> float:
    return sum(
        (float(_field(first, coordinate)) - float(_field(second, coordinate))) ** 2
        for coordinate in ("pos_x", "pos_y", "pos_z")
    )


def _unit_slot(unit: Any, observing_unit: Any) -> list[float]:
    return [
        1.0,  # Slot is occupied
        float(_field(unit, "unit_def_id")),  # Unit type
        _health_fraction(unit),  # Current health / maximum health
        (float(_field(unit, "pos_x")) - float(_field(observing_unit, "pos_x")))
        / POSITION_SCALE,  # Relative X
        (float(_field(unit, "pos_y")) - float(_field(observing_unit, "pos_y")))
        / POSITION_SCALE,  # Relative Y
        (float(_field(unit, "pos_z")) - float(_field(observing_unit, "pos_z")))
        / POSITION_SCALE,  # Relative Z
        float(_field(unit, "los_radius")) / POSITION_SCALE,  # Scaled sight radius
        float(bool(_field(unit, "is_dead"))),  # Dead-state flag
        float(bool(_field(unit, "being_built"))),  # Construction-state flag
    ]


def _nearest_units(units: list[Any], observing_unit: Any, limit: int) -> list[Any]:
    return sorted(
        units,
        key=lambda unit: (
            _distance_squared(unit, observing_unit),
            int(_field(unit, "unit_id")),
        ),
    )[:limit]


def build_r_mappo_observations(
    unit_dictionary: Mapping[str, Any],
    training_team_id: int = 0,
) -> np.ndarray:
    """Convert BAR's agent-to-UnitData dictionary into MAPPO's agent matrix."""
    # Each scalar NumPy array wraps one bound UnitData object.
    units = [_unit_record(value) for value in unit_dictionary.values()]
    # Keep only controlled units and give their rows a stable order.
    controlled_units = sorted(
        (
            unit
            for unit in units
            if int(_field(unit, "team_id")) == training_team_id
        ),
        key=lambda unit: int(_field(unit, "unit_id")),
    )
    observations = np.zeros((len(controlled_units), OBS_DIM), dtype=np.float32)

    for row, unit in enumerate(controlled_units):
        unit_id = int(_field(unit, "unit_id"))
        ally_team_id = int(_field(unit, "ally_team_id"))
        # Team membership determines ally/enemy slots; distance picks their order.
        allies = [
            candidate
            for candidate in units
            if int(_field(candidate, "unit_id")) != unit_id
            and int(_field(candidate, "ally_team_id")) == ally_team_id
        ]
        enemies = [
            candidate
            for candidate in units
            if int(_field(candidate, "ally_team_id")) != ally_team_id
        ]
        nearest_allies = _nearest_units(units=allies, observing_unit=unit, limit=MAX_ALLY_SLOTS)
        nearest_enemies = _nearest_units(units=enemies, observing_unit=unit, limit=MAX_ENEMY_SLOTS)

        features = _self_features(unit)
        for neighbors, slot_count in (
            (nearest_allies, MAX_ALLY_SLOTS),
            (nearest_enemies, MAX_ENEMY_SLOTS),
        ):
            for neighbor in neighbors:
                features.extend(_unit_slot(neighbor, unit))
            # Empty slots stay zero, with their occupied flag therefore false.
            for _ in range(slot_count - len(neighbors)):
                features.extend([0.0] * UNIT_SLOT_FEATURE_DIM)

        observations[row] = np.asarray(features, dtype=np.float32)

    return observations