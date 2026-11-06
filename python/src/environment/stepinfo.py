import logging
import numpy as np
from typing import Any

class StepInfo():
    def __init__(self, action_dim):

        self.episode_reward = 0.0
        self.done = False
        self.terminated = False
        self.truncated = False
        self.step_count = 0

        self.action_counts = np.zeros(
            action_dim,
            dtype=np.int64,
        )
        self.action_trace: list[
            list[int]
        ] = []

        self.episode_won = False
        self.episode_lost = False
        self.final_ally_alive = None
        self.final_enemy_alive = None
        self.enemy_damage_done = 0.0
        self.own_damage_taken = 0.0
        self.last_info_keys: list[str] = []
        self.final_step_info: dict[
            str,
            Any,
        ] = {}

    def receive_prepared(self, prepared):
        self.prepared = prepared

    def update_stepinfo(self, obs_raw: list[Any], episode_number: int):
        return 0