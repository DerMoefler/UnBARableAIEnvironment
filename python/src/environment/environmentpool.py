import json
import logging
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
        self.logger.info("Created Environmentpool")

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

    def step_through(self):
        


