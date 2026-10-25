
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



def _controlled_units(
    unit_dictionary: Mapping[str, Any],
    training_team_id: int,
) -> list[Any]:
    """Return controlled units sorted by their actual unit IDs."""
    units = [
        _unit_record(value)
        for value in unit_dictionary.values()
    ]

    controlled = sorted(
        (
            unit
            for unit in units
            if int(_field(unit, "team_id")) == training_team_id
        ),
        key=lambda unit: int(_field(unit, "unit_id")),
    )

    unit_ids = [
        int(_field(unit, "unit_id"))
        for unit in controlled
    ]

    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError(
            f"Doppelte Unit-IDs im selben Team gefunden: {unit_ids}"
        )

    return controlled



def _health_fraction(unit: Any) -> float:
    """Return health as a fraction of maximum health."""
    max_health = max(
        0.0,
        float(_field(unit, "max_health")),
    )

    if max_health == 0.0:
        return 0.0

    health = float(_field(unit, "health"))

    return float(
        np.clip(health / max_health, 0.0, 1.0)
    )


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
        float(bool(_field(unit, "being_built"))),
    ]


def _distance_squared(
    first: Any,
    second: Any,
) -> float:
    """Squared 3D distance between two units."""
    return sum(
        (
            float(_field(first, coordinate))
            - float(_field(second, coordinate))
        ) ** 2
        for coordinate in ("pos_x", "pos_y", "pos_z")
    )


def _unit_slot(
    unit: Any,
    observing_unit: Any,
) -> list[float]:
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
    """Select nearest units, with a stable tie-breaker."""
    return sorted(
        units,
        key=lambda unit: (
            _distance_squared(unit, observing_unit),
            int(_field(unit, "unit_id")),
        ),
    )[:limit]

def _prepare_obs(
    obs: Any,
    num_agents: int,
    obs_dim: int,
    training_team_id: int = 0,
) -> np.ndarray:
    """
    Converts raw observations to a fixed per-agent float32 array.

    Parameters
    ----------
    obs : Any
        Raw observation data returned by the environment.
    num_agents : int
        Number of agents represented in the result.
    obs_dim : int
        Width of each agent observation.

    Returns
    -------
    observations : np.ndarray
        Array with shape `(num_agents, obs_dim)` and dtype float32.

    Examples
    --------
    >>> _prepare_obs(np.zeros(4), 2, 4).shape
    (2, 4)
    """
    if obs is None:
        return np.zeros((num_agents, obs_dim), dtype=np.float32)

    if isinstance(obs, dict):
        obs = build_r_mappo_observations(
            obs,
            training_team_id=training_team_id,
        )

    arr = np.asarray(obs, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size == obs_dim:
            arr = np.tile(arr[None, :], (num_agents, 1))
        elif arr.size == num_agents * obs_dim:
            arr = arr.reshape(num_agents, obs_dim)
        else:
            fixed = np.zeros((num_agents * obs_dim,), dtype=np.float32)
            n = min(fixed.size, arr.size)
            fixed[:n] = arr.reshape(-1)[:n]
            arr = fixed.reshape(num_agents, obs_dim)

    elif arr.ndim >= 2:
        if arr.shape[0] != num_agents:
            flat = arr.reshape(-1)
            fixed = np.zeros((num_agents * obs_dim,), dtype=np.float32)
            n = min(fixed.size, flat.size)
            fixed[:n] = flat[:n]
            arr = fixed.reshape(num_agents, obs_dim)
        else:
            arr = arr.reshape(num_agents, -1)

            if arr.shape[1] != obs_dim:
                fixed = np.zeros((num_agents, obs_dim), dtype=np.float32)
                n = min(obs_dim, arr.shape[1])
                fixed[:, :n] = arr[:, :n]
                arr = fixed

    return arr.astype(np.float32)

def _prepare_share_obs(
    share_obs: Any,
    obs: np.ndarray,
    num_agents: int,
    obs_dim: int,
) -> np.ndarray:
    """
    Converts raw centralized observations to a fixed float32 vector.

    Parameters
    ----------
    share_obs : Any
        Raw shared observation data returned by the environment.
    obs : np.ndarray
        Prepared per-agent observations used as a fallback.
    num_agents : int
        Number of agents represented in the shared observation.
    obs_dim : int
        Width of the shared observation vector.

    Returns
    -------
    shared_observation : np.ndarray
        Vector with shape `(obs_dim,)` and dtype float32.

    Examples
    --------
    >>> obs = np.zeros((2, 4), dtype=np.float32)
    >>> _prepare_share_obs(None, obs, 2, 4).shape
    (4,)
    """
    if share_obs is None:
        return obs.mean(axis=0).astype(np.float32)

    arr = np.asarray(share_obs, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size == obs_dim:
            return arr.astype(np.float32)

        fixed = np.zeros((obs_dim,), dtype=np.float32)
        n = min(obs_dim, arr.size)
        fixed[:n] = arr.reshape(-1)[:n]
        return fixed

    if arr.ndim >= 2:
        if arr.shape[0] == num_agents:
            arr = arr.reshape(num_agents, -1)

            if arr.shape[1] == obs_dim:
                return arr.mean(axis=0).astype(np.float32)

            fixed = np.zeros((num_agents, obs_dim), dtype=np.float32)
            n = min(obs_dim, arr.shape[1])
            fixed[:, :n] = arr[:, :n]
            return fixed.mean(axis=0).astype(np.float32)

        flat = arr.reshape(-1)
        fixed = np.zeros((obs_dim,), dtype=np.float32)
        n = min(obs_dim, flat.size)
        fixed[:n] = flat[:n]
        return fixed

    return obs.mean(axis=0).astype(np.float32)

def build_r_mappo_observations(
    unit_dictionary: Mapping[str, Any],
    training_team_id: int = 0,
) -> np.ndarray:
    """Build one observation per controlled unit.

    Observation layout:
    - 8 self features
    - 3 nearest enemy slots with 9 features each
    """
    units = [
        _unit_record(value)
        for value in unit_dictionary.values()
    ]

    controlled_units = _controlled_units(
        unit_dictionary,
        training_team_id,
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

        # Keep observation size fixed if fewer than 3 enemies exist.
        while len(features) < OBS_DIM:
            features.extend(
                [0.0] * UNIT_SLOT_FEATURE_DIM
            )

        observations[row] = np.asarray(
            features[:OBS_DIM],
            dtype=np.float32,
        )

    return observations


def build_r_mappo_action_context(
    unit_dictionary: Mapping[str, Any],
    training_team_id: int = 0,
):
    """Return own unit IDs and matching enemy target candidates/features."""
    units = [
        _unit_record(value)
        for value in unit_dictionary.values()
    ]

    controlled_units = _controlled_units(
        unit_dictionary,
        training_team_id,
    )

    own_unit_ids = []

    target_ids = np.full(
        (len(controlled_units), MAX_ENEMY_SLOTS),
        -1,
        dtype=np.int64,
    )

    target_features = np.zeros(
        (
            len(controlled_units),
            MAX_ENEMY_SLOTS,
            UNIT_SLOT_FEATURE_DIM,
        ),
        dtype=np.float32,
    )

    target_mask = np.zeros(
        (len(controlled_units), MAX_ENEMY_SLOTS),
        dtype=bool,
    )

    for row, unit in enumerate(controlled_units):
        own_unit_ids.append(
            int(_field(unit, "unit_id"))
        )

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

        for col, enemy in enumerate(nearest_enemies):
            target_ids[row, col] = int(
                _field(enemy, "unit_id")
            )

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
