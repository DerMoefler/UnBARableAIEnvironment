from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreparedStep:
    """
    Complete fixed-size input for one R-MAPPO environment step.

    All arrays use the same stable controlled-unit slot assignment.
    """

    obs: np.ndarray
    share_obs: np.ndarray
    available_actions: np.ndarray

    own_unit_ids: np.ndarray
    target_ids: np.ndarray
    target_features: np.ndarray
    target_mask: np.ndarray

    active_mask: np.ndarray


class RMAPPOObservationAdapter:
    """
    Converts raw BAR UnitData observations into fixed-size R-MAPPO data.

    Controlled units receive stable slots at episode reset. If a unit
    dies and disappears from shared memory, its original slot remains
    reserved and receives:

    - a zero observation,
    - own_unit_id = -1,
    - active_mask = 0,
    - no valid target,
    - one valid dummy action at action index 0.

    The stable slot assignment is recreated on every episode reset.
    """

    SELF_FEATURE_NAMES = (
        "unit_def_id",
        "health_fraction",
        "pos_x",
        "pos_y",
        "pos_z",
        "los_radius",
        "is_dead",
        "being_built",
    )

    TARGET_FEATURE_NAMES = (
        "occupied",
        "unit_def_id",
        "health_fraction",
        "relative_pos_x",
        "relative_pos_y",
        "relative_pos_z",
        "los_radius",
        "is_dead",
        "being_built",
    )

    def __init__(
        self,
        *,
        num_agents: int,
        action_dim: int,
        training_team_id: int,
        max_enemy_slots: int,
        position_scale: float,
    ) -> None:
        if (
            not isinstance(num_agents, int)
            or isinstance(num_agents, bool)
            or num_agents <= 0
        ):
            raise ValueError(
                "num_agents must be a positive integer."
            )

        if (
            not isinstance(action_dim, int)
            or isinstance(action_dim, bool)
            or action_dim <= 0
        ):
            raise ValueError(
                "action_dim must be a positive integer."
            )

        if (
            not isinstance(max_enemy_slots, int)
            or isinstance(max_enemy_slots, bool)
            or max_enemy_slots <= 0
        ):
            raise ValueError(
                "max_enemy_slots must be a positive integer."
            )

        if float(position_scale) <= 0.0:
            raise ValueError(
                "position_scale must be greater than 0."
            )

        self.num_agents = int(num_agents)
        self.action_dim = int(action_dim)
        self.training_team_id = int(
            training_team_id
        )
        self.max_enemy_slots = int(
            max_enemy_slots
        )
        self.position_scale = float(
            position_scale
        )

        self.agent_unit_ids: tuple[int, ...] | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> RMAPPOObservationAdapter:
        """
        Create an adapter from the central training configuration.
        """
        observation_config = config.get(
            "observation"
        )

        if not isinstance(
            observation_config,
            Mapping,
        ):
            raise ValueError(
                "Configuration must contain an "
                "'observation' object."
            )

        required_root_keys = {
            "number_of_agents",
            "action_dimension",
            "training_team_id",
        }

        missing_root_keys = sorted(
            required_root_keys - config.keys()
        )

        if missing_root_keys:
            raise ValueError(
                "Missing configuration keys: "
                + ", ".join(missing_root_keys)
            )

        required_observation_keys = {
            "max_enemy_slots",
            "position_scale",
        }

        missing_observation_keys = sorted(
            required_observation_keys
            - observation_config.keys()
        )

        if missing_observation_keys:
            raise ValueError(
                "Missing observation configuration keys: "
                + ", ".join(
                    missing_observation_keys
                )
            )

        return cls(
            num_agents=int(
                config["number_of_agents"]
            ),
            action_dim=int(
                config["action_dimension"]
            ),
            training_team_id=int(
                config["training_team_id"]
            ),
            max_enemy_slots=int(
                observation_config[
                    "max_enemy_slots"
                ]
            ),
            position_scale=float(
                observation_config[
                    "position_scale"
                ]
            ),
        )

    @property
    def self_feature_dim(self) -> int:
        return len(self.SELF_FEATURE_NAMES)

    @property
    def target_feature_dim(self) -> int:
        return len(
            self.TARGET_FEATURE_NAMES
        )

    @property
    def obs_dim(self) -> int:
        return (
            self.self_feature_dim
            + self.max_enemy_slots
            * self.target_feature_dim
        )

    def reset_episode(
        self,
        obs_raw: Mapping[str, Any],
    ) -> None:
        """
        Initialize stable controlled-unit slots for a new episode.

        The reset observation must contain exactly num_agents living
        controlled units.
        """
        if not isinstance(obs_raw, Mapping):
            raise TypeError(
                "The reset observation must be a mapping "
                "containing UnitData objects."
            )

        controlled_units = self._controlled_units(
            obs_raw
        )

        unit_ids = tuple(
            int(self._field(unit, "unit_id"))
            for unit in controlled_units
        )

        if len(unit_ids) != self.num_agents:
            raise ValueError(
                "Reset observation must contain exactly "
                f"{self.num_agents} living controlled units, "
                f"but contains {len(unit_ids)}: "
                f"{list(unit_ids)}."
            )

        self.agent_unit_ids = unit_ids

    def prepare(
        self,
        obs_raw: Mapping[str, Any],
        share_obs_raw: Any = None,
        available_actions_raw: Any = None,
    ) -> PreparedStep:
        """
        Prepare all policy inputs from one raw environment snapshot.

        Observation features, target IDs, target features and masks are
        built together so they always share the same agent slots and
        enemy ordering.
        """
        self._require_initialized()

        if not isinstance(obs_raw, Mapping):
            raise TypeError(
                "RMAPPOObservationAdapter.prepare requires "
                "a mapping containing UnitData objects."
            )

        units = self._all_units(obs_raw)

        controlled_by_id = {
            int(self._field(unit, "unit_id")): unit
            for unit in self._controlled_units(
                obs_raw
            )
        }

        obs = np.zeros(
            (
                self.num_agents,
                self.obs_dim,
            ),
            dtype=np.float32,
        )

        own_unit_ids = np.full(
            (self.num_agents,),
            -1,
            dtype=np.int64,
        )

        target_ids = np.full(
            (
                self.num_agents,
                self.max_enemy_slots,
            ),
            -1,
            dtype=np.int64,
        )

        target_features = np.zeros(
            (
                self.num_agents,
                self.max_enemy_slots,
                self.target_feature_dim,
            ),
            dtype=np.float32,
        )

        target_mask = np.zeros(
            (
                self.num_agents,
                self.max_enemy_slots,
            ),
            dtype=np.bool_,
        )

        for row, unit_id in enumerate(
            self.agent_unit_ids
        ):
            controlled_unit = controlled_by_id.get(
                unit_id
            )

            # Die Unit ist tot oder nicht mehr im Shared Memory.
            # Ihr Agentenslot bleibt als Nullzeile bestehen.
            if controlled_unit is None:
                continue

            own_unit_ids[row] = unit_id

            obs[
                row,
                :self.self_feature_dim,
            ] = self._encode_self(
                controlled_unit
            )

            enemies = self._nearest_enemies(
                units,
                controlled_unit,
            )

            for enemy_slot, enemy in enumerate(
                enemies
            ):
                encoded_enemy = self._encode_target(
                    enemy,
                    controlled_unit,
                )

                obs_start = (
                    self.self_feature_dim
                    + enemy_slot
                    * self.target_feature_dim
                )
                obs_end = (
                    obs_start
                    + self.target_feature_dim
                )

                obs[
                    row,
                    obs_start:obs_end,
                ] = encoded_enemy

                target_ids[
                    row,
                    enemy_slot,
                ] = int(
                    self._field(
                        enemy,
                        "unit_id",
                    )
                )

                target_features[
                    row,
                    enemy_slot,
                ] = encoded_enemy

                target_mask[
                    row,
                    enemy_slot,
                ] = True

        active_mask = (
            own_unit_ids >= 0
        ).astype(
            np.float32
        ).reshape(
            self.num_agents,
            1,
        )

        share_obs = self._encode_share_obs(
            share_obs_raw,
            obs,
            active_mask,
        )

        available_actions = (
            self._encode_available_actions(
                available_actions_raw,
                active_mask,
            )
        )

        return PreparedStep(
            obs=obs,
            share_obs=share_obs,
            available_actions=available_actions,
            own_unit_ids=own_unit_ids,
            target_ids=target_ids,
            target_features=target_features,
            target_mask=target_mask,
            active_mask=active_mask,
        )

    def prepare_reward(
        self,
        reward: Any,
    ) -> np.ndarray:
        """
        Convert a scalar or array reward into shape (num_agents, 1).
        """
        if reward is None:
            return np.zeros(
                (self.num_agents, 1),
                dtype=np.float32,
            )

        reward_array = np.asarray(
            reward,
            dtype=np.float32,
        ).reshape(-1)

        if reward_array.size == 1:
            return np.full(
                (self.num_agents, 1),
                float(reward_array[0]),
                dtype=np.float32,
            )

        result = np.zeros(
            (self.num_agents, 1),
            dtype=np.float32,
        )

        count = min(
            self.num_agents,
            reward_array.size,
        )

        result[:count, 0] = reward_array[:count]

        return result

    def _encode_share_obs(
        self,
        share_obs_raw: Any,
        obs: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Build a fixed centralized observation for the critic.

        If the environment supplies no centralized observation, the
        mean of living controlled-unit observations is used.
        """
        if share_obs_raw is not None:
            return self._fit_array(
                share_obs_raw,
                (self.obs_dim,),
            )

        active_rows = (
            active_mask.reshape(-1) > 0.0
        )

        if not active_rows.any():
            return np.zeros(
                (self.obs_dim,),
                dtype=np.float32,
            )

        return obs[
            active_rows
        ].mean(
            axis=0
        ).astype(np.float32)

    def _encode_available_actions(
        self,
        available_actions_raw: Any,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create fixed action masks.

        Living agent slots receive all actions if the environment does
        not supply an explicit mask.

        Dead slots receive only action index 0 as a valid dummy action.
        The PolicyRunner does not send that dummy action to BAR.
        """
        if available_actions_raw is None:
            result = np.ones(
                (
                    self.num_agents,
                    self.action_dim,
                ),
                dtype=np.float32,
            )
        else:
            result = self._fit_array(
                available_actions_raw,
                (
                    self.num_agents,
                    self.action_dim,
                ),
                fill_value=1.0,
            )

        for row, active in enumerate(
            active_mask.reshape(-1)
        ):
            if active <= 0.0:
                result[row] = 0.0
                result[row, 0] = 1.0

        return result.astype(
            np.float32,
            copy=False,
        )

    def _encode_self(
        self,
        unit: Any,
    ) -> np.ndarray:
        """
        Encode one controlled unit.

        The order matches SELF_FEATURE_NAMES.
        """
        result = np.asarray(
            [
                float(
                    self._field(
                        unit,
                        "unit_def_id",
                    )
                ),
                self._health_fraction(unit),
                self._scaled_position(
                    unit,
                    "pos_x",
                ),
                self._scaled_position(
                    unit,
                    "pos_y",
                ),
                self._scaled_position(
                    unit,
                    "pos_z",
                ),
                float(
                    self._field(
                        unit,
                        "los_radius",
                    )
                ) / self.position_scale,
                float(
                    bool(
                        self._field(
                            unit,
                            "is_dead",
                        )
                    )
                ),
                float(
                    bool(
                        self._field(
                            unit,
                            "being_built",
                        )
                    )
                ),
            ],
            dtype=np.float32,
        )

        if result.shape != (
            self.self_feature_dim,
        ):
            raise RuntimeError(
                "Self feature schema does not match "
                f"encoded data: schema={self.self_feature_dim}, "
                f"encoded={result.shape}."
            )

        return result

    def _encode_target(
        self,
        target: Any,
        observing_unit: Any,
    ) -> np.ndarray:
        """
        Encode one enemy relative to one controlled unit.

        The order matches TARGET_FEATURE_NAMES.
        """
        result = np.asarray(
            [
                1.0,
                float(
                    self._field(
                        target,
                        "unit_def_id",
                    )
                ),
                self._health_fraction(target),
                self._relative_position(
                    target,
                    observing_unit,
                    "pos_x",
                ),
                self._relative_position(
                    target,
                    observing_unit,
                    "pos_y",
                ),
                self._relative_position(
                    target,
                    observing_unit,
                    "pos_z",
                ),
                float(
                    self._field(
                        target,
                        "los_radius",
                    )
                ) / self.position_scale,
                float(
                    bool(
                        self._field(
                            target,
                            "is_dead",
                        )
                    )
                ),
                float(
                    bool(
                        self._field(
                            target,
                            "being_built",
                        )
                    )
                ),
            ],
            dtype=np.float32,
        )

        if result.shape != (
            self.target_feature_dim,
        ):
            raise RuntimeError(
                "Target feature schema does not match "
                f"encoded data: schema={self.target_feature_dim}, "
                f"encoded={result.shape}."
            )

        return result

    def _controlled_units(
        self,
        unit_dictionary: Mapping[str, Any],
    ) -> list[Any]:
        """
        Return living controlled units sorted by BAR unit ID.
        """
        controlled_units = sorted(
            (
                unit
                for unit in self._all_units(
                    unit_dictionary
                )
                if int(
                    self._field(
                        unit,
                        "team_id",
                    )
                ) == self.training_team_id
                and self._is_alive(unit)
            ),
            key=lambda unit: int(
                self._field(
                    unit,
                    "unit_id",
                )
            ),
        )

        unit_ids = [
            int(self._field(unit, "unit_id"))
            for unit in controlled_units
        ]

        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError(
                "Duplicate controlled unit IDs: "
                f"{unit_ids}"
            )

        return controlled_units

    def _nearest_enemies(
        self,
        units: list[Any],
        observing_unit: Any,
    ) -> list[Any]:
        """
        Return nearest living enemies with unit ID as tie-breaker.
        """
        ally_team_id = int(
            self._field(
                observing_unit,
                "ally_team_id",
            )
        )

        enemies = [
            candidate
            for candidate in units
            if int(
                self._field(
                    candidate,
                    "ally_team_id",
                )
            ) != ally_team_id
            and self._is_alive(candidate)
        ]

        return sorted(
            enemies,
            key=lambda candidate: (
                self._distance_squared(
                    candidate,
                    observing_unit,
                ),
                int(
                    self._field(
                        candidate,
                        "unit_id",
                    )
                ),
            ),
        )[:self.max_enemy_slots]

    def _health_fraction(
        self,
        unit: Any,
    ) -> float:
        max_health = max(
            0.0,
            float(
                self._field(
                    unit,
                    "max_health",
                )
            ),
        )

        if max_health == 0.0:
            return 0.0

        health = float(
            self._field(
                unit,
                "health",
            )
        )

        return float(
            np.clip(
                health / max_health,
                0.0,
                1.0,
            )
        )

    def _scaled_position(
        self,
        unit: Any,
        coordinate: str,
    ) -> float:
        return (
            float(
                self._field(
                    unit,
                    coordinate,
                )
            )
            / self.position_scale
        )

    def _relative_position(
        self,
        target: Any,
        observing_unit: Any,
        coordinate: str,
    ) -> float:
        return (
            float(
                self._field(
                    target,
                    coordinate,
                )
            )
            - float(
                self._field(
                    observing_unit,
                    coordinate,
                )
            )
        ) / self.position_scale

    def _distance_squared(
        self,
        first: Any,
        second: Any,
    ) -> float:
        return sum(
            (
                float(
                    self._field(
                        first,
                        coordinate,
                    )
                )
                - float(
                    self._field(
                        second,
                        coordinate,
                    )
                )
            ) ** 2
            for coordinate in (
                "pos_x",
                "pos_y",
                "pos_z",
            )
        )

    def _is_alive(
        self,
        unit: Any,
    ) -> bool:
        return (
            not bool(
                self._field(
                    unit,
                    "is_dead",
                )
            )
            and float(
                self._field(
                    unit,
                    "health",
                )
            ) > 0.0
        )

    def _all_units(
        self,
        unit_dictionary: Mapping[str, Any],
    ) -> list[Any]:
        return [
            self._unwrap_unit(value)
            for value in unit_dictionary.values()
        ]

    def _require_initialized(self) -> None:
        if self.agent_unit_ids is None:
            raise RuntimeError(
                "Agent slots are not initialized. "
                "Call reset_episode(obs_raw) after "
                "environment.reset()."
            )

    @staticmethod
    def _unwrap_unit(value: Any) -> Any:
        if (
            isinstance(value, np.ndarray)
            and value.shape == ()
        ):
            return value.item()

        return value

    @staticmethod
    def _field(
        unit: Any,
        name: str,
    ) -> Any:
        if isinstance(unit, Mapping):
            return unit[name]

        return getattr(unit, name)

    @staticmethod
    def _fit_array(
        value: Any,
        shape: tuple[int, ...],
        fill_value: float = 0.0,
    ) -> np.ndarray:
        source = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

        result = np.full(
            int(np.prod(shape)),
            fill_value,
            dtype=np.float32,
        )

        count = min(
            source.size,
            result.size,
        )

        result[:count] = source[:count]

        return result.reshape(shape)