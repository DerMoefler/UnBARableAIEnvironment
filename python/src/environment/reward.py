from dataclasses import dataclass
from logging import config
from typing import Any, Dict, Optional, List
from src.environment.episode_outcome import EpisodeOutcome
from src.environment.unit_data_memory import UnitMemory
import bar_ai

import numpy as np

@dataclass
class RewardConfig:
#Default reward configuration values
    damage_enemy_coef: float 
    damage_own_coef: float 
    enemy_kill: float 
    own_death: float 
    win: float 
    loss: float 
    time_penalty: float 
    clip_min: float 
    clip_max: float 


class RewardCalculator:
    """Calculates training rewards from per-step team statistics."""
    

    def __init__(self, training_team_id: int, config: RewardConfig) -> None:
        self.training_team_id = training_team_id
        self.config = config
        self.prev_own_health_sum = 0.0
        self.prev_enemy_health_sum = 0.0
        self.prev_own_alive_count = 0
        self.prev_enemy_alive_count = 0
        self.initialized = False
        self.last_info: Dict[str, Any] = {}
        self.unit_memory = UnitMemory(self.training_team_id)

    def reset(self, units: List[bar_ai.UnitData]) -> None:
        """Set the initial reward baseline for an episode."""

        self.unit_memory.reset(units)
        stats = self._get_team_stats(units)

        self.prev_own_health_sum = stats["own_health_sum"]
        self.prev_enemy_health_sum = stats["enemy_health_sum"]
        self.prev_own_alive_count = stats["own_alive_count"]
        self.prev_enemy_alive_count = stats["enemy_alive_count"]
        self.initialized = not (
            self.prev_own_alive_count == 0 and self.prev_enemy_alive_count == 0
        )
        self.last_info = self._make_info(stats, self.initialized)

    def calculate(
        self,
        units: List[bar_ai.UnitData],
        *,
        terminated: bool = False,
        truncated: bool = False,
        outcome: Optional[EpisodeOutcome] = None,
    ) -> float:
        """Calculate a clipped reward and retain its component diagnostics."""

        self.unit_memory.update(units)
        stats = self._get_team_stats(units)

        own_health_sum = stats["own_health_sum"]
        enemy_health_sum = stats["enemy_health_sum"]
        own_alive_count = stats["own_alive_count"]
        enemy_alive_count = stats["enemy_alive_count"]

        if (own_alive_count == 0 and enemy_alive_count == 0 and not terminated and not truncated):
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

        # Determine episode outcome reward
        win_bonus = 0.0
        loss_penalty = 0.0

        if terminated or truncated:
            if outcome == EpisodeOutcome.WON:
                win_bonus = config.win
 
            elif outcome == EpisodeOutcome.LOST:
                loss_penalty = config.loss

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

    def _get_team_stats(
            self,
            current_units: List[bar_ai.UnitData],
        ) -> Dict[str, Any]:
    
            own_alive = [
                unit
                for unit in current_units
                if int(unit.team_id) == self.training_team_id
                and not bool(unit.is_dead)
                and float(unit.health) > 0.0
            ]
    
            remembered_enemy_alive = self.unit_memory.get_alive_units()
    
            return {
                "own_health_sum": float(sum(
                    max(0.0, float(unit.health))
                    for unit in own_alive
                )),
                "enemy_health_sum": float(sum(
                    max(0.0, float(unit.health))
                    for unit in remembered_enemy_alive
                )),
                "own_alive_count": len(own_alive),
                "enemy_alive_count": len(remembered_enemy_alive),
            }
    
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