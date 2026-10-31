import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.environment.bar_environment import BAR_Environment

from src.utils.logger_factory import get_file_logger

class EnvironmentPool:
    data: dict[str, Any]
    def __init__(self, config_path: Path):
        with open(config_path,"r") as config_data:
            self.data = json.load(config_data)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.logger = get_file_logger("pool", Path(f"src/train/experiments/{self.data["run_name"]}/{self.data["run_name"]}_pool_{timestamp}.log"))
        self.logger.info("Created Environmentpool")

    def initialize_environements(self):
        logging.info("Initializing Environments")
        for i in range (self.data["number_of_environments"]):
            return True

    def close_environements(self):
        return False
    





#steap trhough



