#!/usr/bin/env python3

# Collection of helpers / utility

from dataclasses import dataclass
import json
import time

from collections.abc import Callable
from typing import Any
from pathlib import Path
import copy

def wait_until(condition: Callable[[], bool], timeout: float = 30.0, interval: float = 0.05) -> None:
    """
    Simple helper to sleep a thread, repeatedly checking if the condition has been satisfied yet.
    """
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if condition():
            return

        time.sleep(interval)

    raise TimeoutError("Condition was not satisfied before timeout")

@dataclass(frozen=True)
class IntegrationScenario:
    name: str
    self_play: bool = False
    number_of_agents: int = 3
    number_of_enemies: int = 3
    number_of_engines: int = 1
    maximum_steps: int = 512

    @property
    def expected_team_count(self) -> int:
        return 2 if self.self_play else 1

def load_base_config() -> dict[str, Any]:
    config_path = Path(
        "src/train/experiments/3v3Test/"
        "3v3Test.json"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Base training configuration not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(
            config_file
        )

    if not isinstance(config, dict):
        raise ValueError(
            "Base training configuration must "
            "contain a JSON object."
        )

    return config


def create_scenario_config(
    *,
    base_config: dict[str, Any],
    scenario: IntegrationScenario,
    run_name: str,
) -> dict[str, Any]:
    config = copy.deepcopy(
        base_config
    )

    config["run_name"] = run_name
    config["device"] = "cpu"

    config["number_of_agents"] = (
        scenario.number_of_agents
    )

    config["number_of_enemies"] = (
        scenario.number_of_enemies
    )

    config["self_play"] = (
        scenario.self_play
    )

    config["action_dimension"] = 5

    adapter_config = config.setdefault(
        "adapter",
        {},
    )

    adapter_config["position_scale"] = float(
        adapter_config.get(
            "position_scale",
            1000.0,
        )
    )

    engine_config = config.setdefault(
        "engine",
        {},
    )

    engine_config["mode"] = "training"
    engine_config["unit"] = engine_config.get(
        "unit",
        "armpw",
    )
    engine_config["spawn_spacing"] = int(
        engine_config.get(
            "spawn_spacing",
            80,
        )
    )

    ai_options = engine_config.setdefault(
        "ai_options",
        {},
    )

    ai_options[
        "update_interval_frames"
    ] = int(
        ai_options.get(
            "update_interval_frames",
            2,
        )
    )

    ai_options[
        "move_step_size"
    ] = float(
        ai_options.get(
            "move_step_size",
            100.0,
        )
    )

    ai_options[
        "first_update_frame"
    ] = int(
        ai_options.get(
            "first_update_frame",
            2,
        )
    )

    training_config = config.setdefault(
        "training",
        {},
    )

    training_config.update(
        {
            "continue_training": False,

            # Der Buffer speichert vollständige Übergänge
            # eines Teams, nicht einzelne Player-Turns.
            "buffer_size": max(
                4,
                scenario.maximum_steps,
            ),

            "mini_batch": 1,
            "episodes": 1,
            "number_of_environments": 1,
            "number_of_engines": scenario.number_of_engines,
            "save_every_x_episode": 1,
            "maximum_steps": (
                scenario.maximum_steps
            ),
        }
    )

    mappo_config = training_config.setdefault(
        "mappo",
        {},
    )

    mappo_config.update(
        {
            "learning_rate": 1e-4,
            "gamma": 0.99,
        }
    )

    logging_config = config.setdefault(
        "logging",
        {},
    )

    logging_config["level"] = "INFO"

    module_logging = logging_config.setdefault(
        "modules",
        {},
    )

    module_logging.update(
        {
            "src.environment.bar_environment": (
                "INFO"
            ),
            "src.environment.engine_session": (
                "INFO"
            ),
            "src.environment.grpc_server": (
                "INFO"
            ),
        }
    )

    return config


def write_config(
    *,
    config: dict[str, Any],
    destination: Path,
) -> Path:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            config,
            config_file,
            indent=4,
        )

    return destination