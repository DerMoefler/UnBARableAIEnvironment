from concurrent.futures import ThreadPoolExecutor, wait
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any

from src.environment.bar_environment import BAR_Environment


class EnvironmentPool:
    def __init__(self, config_path: Path):
        with open(config_path,"r") as config_data:
            self.data = json.load(config_data)
        self.path = config_path
        self.environments: list[BAR_Environment] = []
        self.logger = logging.getLogger(__name__)
        self._step_executor: ThreadPoolExecutor | None = None
        self.logger.info("Created Environmentpool")

    def initialize_environements(self):
        logging.info("Initializing Environments")
        self.environments = []
        for i in range(self.data["number_of_environments"]):
            self.environments.append(BAR_Environment(training_team_id=self.data["training_team_id"], environment_id=i, config_path=self.path))

    def close_environements(self):
        if self._step_executor is not None:
            self._step_executor.shutdown(wait=True)
            self._step_executor = None

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

    def reset_environments(self) -> list[Any]:
        if not self.environments:
            raise RuntimeError("No environments initialized")
        observations = []

        for environment in self.environments:
            obs_raw = environment.reset()
            observations.append(obs_raw)

        return observations

    def step_environments(
        self,
        actions_by_environment: list[list[Any]],
    ) -> list[Any]:
        if not self.environments:
            raise RuntimeError("No environments initialized")
        if len(actions_by_environment) != len(self.environments):
            raise ValueError(
                "Expected one action list per environment: "
                f"got {len(actions_by_environment)} action lists for "
                f"{len(self.environments)} environments."
            )

        if self._step_executor is None:
            self._step_executor = ThreadPoolExecutor(
                max_workers=len(self.environments),
                thread_name_prefix="bar-environment-step",
            )

        futures = [
            self._step_executor.submit(
                environment.step,
                actions,
            )
            for environment, actions in zip(
                self.environments,
                actions_by_environment,
                strict=True,
            )
        ]
        wait(futures)

        return [
            future.result()
            for future in futures
        ]


