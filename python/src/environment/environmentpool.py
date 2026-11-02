import json
import logging
from pathlib import Path
from typing import Any

from src.environment.bar_environment import BAR_Environment


class EnvironmentPool:
    data: dict[str, Any]
    def __init__(self, config_path: Path):
        with open(config_path,"r") as config_data:
            self.data = json.load(config_data)

        self.logger = logging.getLogger(__name__)
        self.logger.info("Created Environmentpool")

    def initialize_environements(self):
        logging.info("Initializing Environments")
        for i in range (self.data["number_of_environments"]):
            return True

    def close_environements(self):
        return False
    





#steap trhough


