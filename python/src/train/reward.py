from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RewardConfig:
    training_team_id: int = 0
    damage_enemy_coef: float = 0.01
    damage_own_coef: float = 0.01
    enemy_kill: float = 1.0
    own_death: float = 1.0
    win: float = 5.0
    loss: float = 5.0
    time_penalty: float = 0.001
    clip_min: float = -10.0
    clip_max: float = 10.0


class RewardCalculator:
    """Calculates training rewards from per-step team statistics."""

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config if config is not None else RewardConfig()
        self.prev_own_health_sum = 0.0
        self.prev_enemy_health_sum = 0.0
        self.prev_own_alive_count = 0
        self.prev_enemy_alive_count = 0
        self.initialized = False
        self.last_info: Dict[str, Any] = {}

    def reset(self, stats: Dict[str, Any]) -> None:
        """Set the initial reward baseline for an episode."""
        self.prev_own_health_sum = stats["own_health_sum"]
        self.prev_enemy_health_sum = stats["enemy_health_sum"]
        self.prev_own_alive_count = stats["own_alive_count"]
        self.prev_enemy_alive_count = stats["enemy_alive_count"]
        self.initialized = not (
            self.prev_own_alive_count == 0 and self.prev_enemy_alive_count == 0
        )
        self.last_info = self._make_info(stats, self.initialized)

    def calculate(self, stats: Dict[str, Any]) -> float:
        """Calculate a clipped reward and retain its component diagnostics."""
        own_health_sum = stats["own_health_sum"]
        enemy_health_sum = stats["enemy_health_sum"]
        own_alive_count = stats["own_alive_count"]
        enemy_alive_count = stats["enemy_alive_count"]

        if own_alive_count == 0 and enemy_alive_count == 0:
            self.initialized = False
            self.last_info = self._make_info(stats, False)
            return 0.0

        if not self.initialized:
            self.prev_own_health_sum = own_health_sum
            self.prev_enemy_health_sum = enemy_health_sum
            self.prev_own_alive_count = own_alive_count
            self.prev_enemy_alive_count = enemy_alive_count
            self.initialized = True
            self.last_info = self._make_info(stats, True)
            return 0.0

        enemy_damage_done = max(0.0, self.prev_enemy_health_sum - enemy_health_sum)
        own_damage_taken = max(0.0, self.prev_own_health_sum - own_health_sum)
        enemy_kills = max(0, self.prev_enemy_alive_count - enemy_alive_count)
        own_deaths = max(0, self.prev_own_alive_count - own_alive_count)

        config = self.config
        reward = (
            config.damage_enemy_coef * enemy_damage_done
            - config.damage_own_coef * own_damage_taken
            + config.enemy_kill * enemy_kills
            - config.own_death * own_deaths
        )

        win_bonus = config.win if enemy_alive_count == 0 and own_alive_count > 0 else 0.0
        loss_penalty = config.loss if own_alive_count == 0 and enemy_alive_count > 0 else 0.0
        reward += win_bonus - loss_penalty - config.time_penalty

        raw_reward = float(reward)
        clipped_reward = float(np.clip(raw_reward, config.clip_min, config.clip_max))

        self.prev_own_health_sum = own_health_sum
        self.prev_enemy_health_sum = enemy_health_sum
        self.prev_own_alive_count = own_alive_count
        self.prev_enemy_alive_count = enemy_alive_count
        self.last_info = self._make_info(
            stats,
            True,
            enemy_damage_done=enemy_damage_done,
            own_damage_taken=own_damage_taken,
            enemy_kills=enemy_kills,
            own_deaths=own_deaths,
            win_bonus=win_bonus,
            loss_penalty=loss_penalty,
            time_penalty=config.time_penalty,
            raw_reward=raw_reward,
            clipped_reward=clipped_reward,
        )
        return clipped_reward

    @staticmethod
    def _make_info(
        stats: Dict[str, Any],
        initialized: bool,
        enemy_damage_done: float = 0.0,
        own_damage_taken: float = 0.0,
        enemy_kills: int = 0,
        own_deaths: int = 0,
        win_bonus: float = 0.0,
        loss_penalty: float = 0.0,
        time_penalty: float = 0.0,
        raw_reward: float = 0.0,
        clipped_reward: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "reward_state_initialized": initialized,
            "own_health_sum": stats["own_health_sum"],
            "enemy_health_sum": stats["enemy_health_sum"],
            "own_alive_count": stats["own_alive_count"],
            "enemy_alive_count": stats["enemy_alive_count"],
            "enemy_damage_done": float(enemy_damage_done),
            "own_damage_taken": float(own_damage_taken),
            "enemy_kills": int(enemy_kills),
            "own_deaths": int(own_deaths),
            "win_bonus": float(win_bonus),
            "loss_penalty": float(loss_penalty),
            "time_penalty": float(time_penalty),
            "raw_reward": float(raw_reward),
            "clipped_reward": float(clipped_reward),
        }