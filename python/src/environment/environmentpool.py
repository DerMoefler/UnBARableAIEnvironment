from concurrent.futures import ThreadPoolExecutor, wait
from concurrent.futures import ThreadPoolExecutor, wait
import json
import logging
import numpy as np
import numpy as np
from pathlib import Path
from typing import Any

from src.environment.bar_environment import BAR_Environment


class EnvironmentPool:
    def __init__(self, config_path: Path):
        with open(config_path,"r") as config_data:
            self.data = json.load(config_data)
        self.maximum_steps = int(self.data.get("maximum_steps", 512))
        if self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be greater than zero.")
        self.maximum_steps = int(self.data.get("maximum_steps", 512))
        if self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be greater than zero.")
        self.path = config_path
        self.environments: list[BAR_Environment] = []
        self.logger = logging.getLogger(__name__)
        self._step_executor: ThreadPoolExecutor | None = None
        self._step_executor: ThreadPoolExecutor | None = None
        self.logger.info("Created Environmentpool")

    def initialize_environements(self):
        logging.info("Initializing Environments")
        self.environments = []
        for i in range(self.data["training"]["number_of_environments"]):
            self.environments.append(BAR_Environment.from_config(self.path, environment_id=i))

    def close_environements(self):
        if self._step_executor is not None:
            self._step_executor.shutdown(wait=True)
            self._step_executor = None

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

    def reset_environment(self, env_id: int) -> Any:
        if not self.environments:
            raise RuntimeError("No environments initialized")
        if not 0 <= env_id < len(self.environments):
            raise IndexError(f"Environment ID {env_id} is out of range for {len(self.environments)} environments.")

        return self.environments[env_id].reset()

    def step_environments(
        self,
        actions_by_environment: list[list[Any]],
    ) -> list[Any | None]:
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
            (
                None
                if (
                    environment.episode_step >= self.maximum_steps
                    or getattr(
                        getattr(environment, "stepinfo", None),
                        "done",
                        False,
                    )
                )
                else self._step_executor.submit(
                    environment.step,
                    actions,
                )
            )
            for environment, actions in zip(
                self.environments,
                actions_by_environment,
                strict=True,
            )
        ]
        wait([
            future
            for future in futures
            if future is not None
        ])

        return [
            future.result() if future is not None else None
            for future in futures
        ]

    def check_done(self):
        for i in range(len(self.environments)):
            if self.environments[i].stepinfo.done == False:
                return False
        return True
    def reset_environment(self, env_id: int) -> Any:
        if not self.environments:
            raise RuntimeError("No environments initialized")
        if not 0 <= env_id < len(self.environments):
            raise IndexError(f"Environment ID {env_id} is out of range for {len(self.environments)} environments.")

        return self.environments[env_id].reset()

    def step_environments(
        self,
        actions_by_environment: list[list[Any]],
    ) -> list[Any | None]:
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
            (
                None
                if (
                    environment.episode_step >= self.maximum_steps
                    or getattr(
                        getattr(environment, "stepinfo", None),
                        "done",
                        False,
                    )
                )
                else self._step_executor.submit(
                    environment.step,
                    actions,
                )
            )
            for environment, actions in zip(
                self.environments,
                actions_by_environment,
                strict=True,
            )
        ]
        wait([
            future
            for future in futures
            if future is not None
        ])

        return [
            future.result() if future is not None else None
            for future in futures
        ]

    def check_done(self):
        for i in range(len(self.environments)):
            if self.environments[i].stepinfo.done == False:
                return False
        return True

