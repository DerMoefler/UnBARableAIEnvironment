import json
import logging
import numpy as np
from pathlib import Path
from typing import Any

from src.environment.bar_environment import BAR_Environment
from src.train.mappo.obs_r_mappo import RMAPPOObservationAdapter


class EnvironmentPool:
    def __init__(self, config_path: Path):
        with open(config_path,"r") as config_data:
            self.data = json.load(config_data)
        self.path = config_path
        self.environments: list[BAR_Environment] = []
        self.logger = logging.getLogger(__name__)
        self.logger.info("Created Environmentpool")
        self.adapters: list[RMAPPOObservationAdapter] = []

    def initialize_environements(self):
        logging.info("Initializing Environments")
        self.environments = []
        for i in range(self.data["number_of_environments"]):
            self.environments.append(BAR_Environment(training_team_id=self.data["training_team_id"], environment_id=i, config_path=self.path))

    def close_environements(self):
        while self.environments:
            environment = self.environments[-1]
            try:
                environment.close()
                self.logger.debug("env.close() done")
            except Exception as close_err:
                self.logger.warning("env.close(), error: %s", close_err)
            finally:
                self.environments.pop()
                self.logger.debug("removed environment from list")

    def create_adapters(self):
        adapters: list[RMAPPOObservationAdapter] = []
        for i in range(len(self.environments)):
            adapters.append(RMAPPOObservationAdapter.from_config(self.data))

            self.logger.info(
                        "Created observation adapter: %s"
                        "obs_dim=%d target_feature_dim=%d "
                        "max_enemy_slots=%d",
                        i,
                        adapters[i].obs_dim,
                        adapters[i].target_feature_dim,
                        adapters[i].max_enemy_slots,
                    )
 
        return adapters
    
    def prepare_step_loop(self, obs_raw: list[Any], episode_number: int):
        for i in range(len(self.adapters)):
            self.adapters[i].reset_episode(obs_raw[i])

            prepared = self.adapters[i].prepare(
                obs_raw=obs_raw[i],
            )

            self.logger.debug(
                "Episode %d stable unit slots: %s",
                episode_number,
                self.adapters[i].agent_unit_ids,
            )

            episode_reward = 0.0
            done = False
            terminated = False
            truncated = False
            step_count = 0

            action_counts = np.zeros(
                self.adapters[i].action_dim,
                dtype=np.int64,
            )
            action_trace: list[
                list[int]
            ] = []

            episode_won = False
            episode_lost = False
            final_ally_alive = None
            final_enemy_alive = None
            enemy_damage_done = 0.0
            own_damage_taken = 0.0
            last_info_keys: list[str] = []
            final_step_info: dict[
                str,
                Any,
            ] = {}

    def reset_environments(self) -> list[Any]:
        if not self.environments:
            raise RuntimeError("No environments initialized")
        observations = []

        for environment in self.environments:
            obs_raw = environment.reset()
            observations.append(obs_raw)

        return observations

    def step_environments(self):
        return 0


