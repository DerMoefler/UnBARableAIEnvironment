from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class StepInfo:
    def __init__(self, action_dim: int) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be greater than zero.")

        self.episode_number: int | None = None
        self.episode_reward = 0.0
        self.done = False
        self.terminated = False
        self.truncated = False
        self.step_count = 0

        self.action_counts = np.zeros(
            action_dim,
            dtype=np.int64,
        )
        self.action_trace: list[list[int]] = []

        self.episode_won = False
        self.episode_lost = False
        self.final_ally_alive: int | None = None
        self.final_enemy_alive: int | None = None
        self.enemy_damage_done = 0.0
        self.own_damage_taken = 0.0
        self.last_info_keys: list[str] = []
        self.final_step_info: dict[str, Any] = {}
        self.prepared: Any | None = None

    def receive_prepared(self, prepared: Any) -> None:
        self.prepared = prepared

    def update_stepinfo(
        self,
        *,
        episode_number: int,
        action_types: Sequence[int] | np.ndarray,
        reward: float | np.ndarray,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        active_mask: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if episode_number <= 0:
            raise ValueError("episode_number must be greater than zero.")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise TypeError("terminated and truncated must be bool values.")
        if terminated and truncated:
            raise ValueError("An episode cannot be terminated and truncated simultaneously.")
        if not isinstance(info, Mapping):
            raise TypeError("info must be a mapping.")
        if self.done:
            raise RuntimeError("Cannot update StepInfo after the episode has ended.")
        if (
            self.episode_number is not None
            and self.episode_number != episode_number
        ):
            raise ValueError(
                f"StepInfo belongs to episode {self.episode_number}, "
                f"not episode {episode_number}."
            )

        actions = [int(action) for action in action_types]
        if not actions:
            raise ValueError("action_types must contain at least one action.")
        if any(action < 0 or action >= len(self.action_counts) for action in actions):
            raise ValueError(
                f"action_types must be in the range [0, {len(self.action_counts) - 1}]."
            )

        if active_mask is None:
            active = np.ones(len(actions), dtype=bool)
        else:
            active = np.asarray(active_mask).reshape(-1) > 0.0
            if len(active) != len(actions):
                raise ValueError(
                    "active_mask must contain one value per action."
                )

        reward_values = np.asarray(reward, dtype=np.float64)
        if reward_values.size == 0:
            raise ValueError("reward must contain at least one value.")
        step_reward = float(reward_values.mean())

        self.episode_number = episode_number
        self.action_trace.append(actions)
        for action, is_active in zip(actions, active, strict=True):
            if is_active:
                self.action_counts[action] += 1

        self.episode_reward += step_reward
        self.step_count += 1
        self.terminated = terminated
        self.truncated = truncated
        self.done = terminated or truncated

        outcome = info.get("episode_outcome")
        self.episode_won = self.episode_won or outcome == "won"
        self.episode_lost = self.episode_lost or outcome == "lost"

        own_alive_count = info.get("own_alive_count")
        if own_alive_count is not None:
            self.final_ally_alive = int(own_alive_count)
            self.episode_lost = self.episode_lost or self.final_ally_alive <= 0

        enemy_alive_count = info.get("enemy_alive_count")
        if enemy_alive_count is not None:
            self.final_enemy_alive = int(enemy_alive_count)
            self.episode_won = self.episode_won or self.final_enemy_alive <= 0

        reward_info = info.get("reward_info")
        if isinstance(reward_info, Mapping):
            self.enemy_damage_done += float(
                reward_info.get("enemy_damage_done", 0.0)
            )
            self.own_damage_taken += float(
                reward_info.get("own_damage_taken", 0.0)
            )

        self.last_info_keys = sorted(info.keys())
        self.final_step_info = dict(info)
