
from collections.abc import Mapping
from typing import Any

import numpy as np


# One self record and three nearest enemy slots.
SELF_FEATURE_DIM = 8
UNIT_SLOT_FEATURE_DIM = 9
MAX_ENEMY_SLOTS = 3

POSITION_SCALE = 1000.0

OBS_DIM = SELF_FEATURE_DIM + MAX_ENEMY_SLOTS * UNIT_SLOT_FEATURE_DIM


def _unit_record(value: Any) -> Any:
    """Unwrap scalar NumPy arrays containing UnitData objects."""
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _field(unit: Any, name: str) -> Any:
    """Read a field from either a dictionary or an object."""
    if isinstance(unit, Mapping):
        return unit[name]
    return getattr(unit, name)


def _health_fraction(unit: Any) -> float:
    """Return health as a fraction of maximum health."""
    max_health = max(0.0, float(_field(unit, "max_health")))

    if max_health == 0.0:
        return 0.0

    health = float(_field(unit, "health"))
    return float(np.clip(health / max_health, 0.0, 1.0))


def _self_features(unit: Any) -> list[float]:
    """Features describing the controlled unit."""
    return [
        float(_field(unit, "unit_def_id")),
        _health_fraction(unit),
        float(_field(unit, "pos_x")) / POSITION_SCALE,
        float(_field(unit, "pos_y")) / POSITION_SCALE,
        float(_field(unit, "pos_z")) / POSITION_SCALE,
        float(_field(unit, "los_radius")) / POSITION_SCALE,
        float(bool(_field(unit, "is_dead"))),
        float(bool(_field(unit, "being_built")),
        ),
    ]


def _distance_squared(first: Any, second: Any) -> float:
    """Squared 3D distance between two units."""
    return sum(
        (
            float(_field(first, coordinate))
            - float(_field(second, coordinate))
        ) ** 2
        for coordinate in ("pos_x", "pos_y", "pos_z")
    )


def _unit_slot(unit: Any, observing_unit: Any) -> list[float]:
    """Features describing an enemy relative to the controlled unit."""
    return [
        1.0,  # Slot occupied
        float(_field(unit, "unit_def_id")),
        _health_fraction(unit),
        (
            float(_field(unit, "pos_x"))
            - float(_field(observing_unit, "pos_x"))
        ) / POSITION_SCALE,
        (
            float(_field(unit, "pos_y"))
            - float(_field(observing_unit, "pos_y"))
        ) / POSITION_SCALE,
        (
            float(_field(unit, "pos_z"))
            - float(_field(observing_unit, "pos_z"))
        ) / POSITION_SCALE,
        float(_field(unit, "los_radius")) / POSITION_SCALE,
        float(bool(_field(unit, "is_dead"))),
        float(bool(_field(unit, "being_built"))),
    ]


def _nearest_units(
    units: list[Any],
    observing_unit: Any,
    limit: int,
) -> list[Any]:
    """Select the nearest units, with a stable tie-breaker."""
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
    """Build one observation per controlled unit.

    Observation layout:
    - 8 self features
    - 3 nearest enemy slots with 9 features each
    """
    units = [_unit_record(value) for value in unit_dictionary.values()]

    controlled_units = sorted(
        (
            unit for unit in units
            if int(_field(unit, "team_id")) == training_team_id
        ),
        key=lambda unit: int(_field(unit, "unit_id")),
    )

    observations = np.zeros(
        (len(controlled_units), OBS_DIM),
        dtype=np.float32,
    )

    for row, unit in enumerate(controlled_units):
        features = list(_self_features(unit))

        ally_team_id = int(_field(unit, "ally_team_id"))
        enemies = [
            candidate
            for candidate in units
            if int(_field(candidate, "ally_team_id")) != ally_team_id
        ]

        nearest_enemies = _nearest_units(
            enemies,
            unit,
            MAX_ENEMY_SLOTS,
        )

        for enemy in nearest_enemies:
            features.extend(_unit_slot(enemy, unit))

        # Keep the observation size fixed if fewer than 3 enemies exist.
        while len(features) < OBS_DIM:
            features.extend([0.0] * UNIT_SLOT_FEATURE_DIM)

        observations[row] = np.asarray(features[:OBS_DIM], dtype=np.float32)

    return observations

def build_r_mappo_action_context(
    unit_dictionary: Mapping[str, Any],
    training_team_id: int = 0,
):
    """Return own unit IDs and matching enemy target candidates/features."""
    units = [_unit_record(value) for value in unit_dictionary.values()]

    controlled_units = sorted(
        (
            unit for unit in units
            if int(_field(unit, "team_id")) == training_team_id
        ),
        key=lambda unit: int(_field(unit, "unit_id")),
    )

    own_unit_ids = []
    target_ids = np.full(
        (len(controlled_units), MAX_ENEMY_SLOTS),
        -1,
        dtype=np.int64,
    )
    target_features = np.zeros(
        (len(controlled_units), MAX_ENEMY_SLOTS, UNIT_SLOT_FEATURE_DIM),
        dtype=np.float32,
    )
    target_mask = np.zeros(
        (len(controlled_units), MAX_ENEMY_SLOTS),
        dtype=bool,
    )

    for row, unit in enumerate(controlled_units):
        own_unit_ids.append(int(_field(unit, "unit_id")))
        ally_team_id = int(_field(unit, "ally_team_id"))

        enemies = [
            candidate for candidate in units
            if int(_field(candidate, "ally_team_id")) != ally_team_id
        ]
        nearest_enemies = _nearest_units(
            enemies,
            unit,
            MAX_ENEMY_SLOTS,
        )

        for col, enemy in enumerate(nearest_enemies):
            target_ids[row, col] = int(_field(enemy, "unit_id"))
            target_features[row, col] = np.asarray(
                _unit_slot(enemy, unit),
                dtype=np.float32,
            )
            target_mask[row, col] = True

    return (
        own_unit_ids,
        target_ids,
        target_features,
        target_mask,
    )