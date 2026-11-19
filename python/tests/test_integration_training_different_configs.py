from __future__ import annotations

import copy
import json
import shutil
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from src.train import training


@dataclass(frozen=True)
class IntegrationScenario:
    name: str
    self_play: bool
    number_of_agents: int
    number_of_enemies: int
    maximum_steps: int

    @property
    def expected_team_count(self) -> int:
        return 2 if self.self_play else 1


SCENARIOS = (
    IntegrationScenario(
        name="normal_3v3",
        self_play=False,
        number_of_agents=3,
        number_of_enemies=3,
        maximum_steps=2,
    ),
    IntegrationScenario(
        name="normal_3v5",
        self_play=False,
        number_of_agents=3,
        number_of_enemies=5,
        maximum_steps=2,
    ),
    IntegrationScenario(
        name="selfplay_3v3",
        self_play=True,
        number_of_agents=3,
        number_of_enemies=3,
        maximum_steps=2,
    ),
    IntegrationScenario(
        name="selfplay_3v5",
        self_play=True,
        number_of_agents=3,
        number_of_enemies=5,
        maximum_steps=2,
    ),
)


def _load_base_config() -> dict[str, Any]:
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


def _create_scenario_config(
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
            "number_of_engines": 1,
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


def _write_config(
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


@pytest.mark.integration
@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=lambda scenario: scenario.name,
)
def test_real_bar_training_scenarios(
    scenario: IntegrationScenario,
    tmp_path: Path,
):
    """
    Full integration test.

    This test starts the real BAR/Recoil engine and uses:

    - the real EnvironmentPool,
    - the real BAR_Environment,
    - real gRPC servers,
    - real shared-memory segments,
    - the real observation adapters,
    - the real shared MAPPO policy,
    - real RolloutBuffers,
    - a real combined PPO update,
    - real checkpoint serialization.

    No environment or engine component is simulated.
    """
    torch.set_num_threads(1)

    unique_suffix = uuid.uuid4().hex[:8]

    run_name = (
        f"integration_{scenario.name}_"
        f"{unique_suffix}"
    )

    experiment_directory = (
        Path("src/train/experiments")
        / run_name
    )

    config_path = (
        tmp_path
        / scenario.name
        / "training_config.json"
    )

    base_config = _load_base_config()

    scenario_config = (
        _create_scenario_config(
            base_config=base_config,
            scenario=scenario,
            run_name=run_name,
        )
    )

    _write_config(
        config=scenario_config,
        destination=config_path,
    )

    train_multiple_calls: list[
        dict[str, Any]
    ] = []

    original_train_multiple = (
        training.R_MAPPO.train_multiple
    )

    def tracked_train_multiple(
        trainer,
        buffers,
        update_actor=True,
    ):
        buffer_list = list(
            buffers
        )

        train_multiple_calls.append(
            {
                "buffer_count": len(
                    buffer_list
                ),
                "buffer_steps": [
                    int(buffer.step)
                    for buffer in buffer_list
                ],
                "buffer_agent_counts": [
                    int(buffer.num_agents)
                    for buffer in buffer_list
                ],
                "update_actor": bool(
                    update_actor
                ),
            }
        )

        return original_train_multiple(
            trainer,
            buffer_list,
            update_actor=update_actor,
        )

    try:
        with patch.object(
            training.R_MAPPO,
            "train_multiple",
            tracked_train_multiple,
        ):
            training.main(
                config_path=config_path
            )

        # --------------------------------------------------------
        # Combined PPO assertions
        # --------------------------------------------------------

        assert len(
            train_multiple_calls
        ) == 1, (
            "Expected exactly one combined PPO update "
            "for the episode, received "
            f"{len(train_multiple_calls)}."
        )

        training_call = (
            train_multiple_calls[0]
        )

        assert (
            training_call["update_actor"]
            is True
        )

        assert (
            training_call["buffer_count"]
            == scenario.expected_team_count
        ), (
            f"Scenario {scenario.name}: expected "
            f"{scenario.expected_team_count} non-empty "
            f"team buffer(s), received "
            f"{training_call['buffer_count']}. "
            f"Buffer steps: "
            f"{training_call['buffer_steps']}"
        )

        assert all(
            step_count > 0
            for step_count
            in training_call["buffer_steps"]
        ), (
            "Every buffer passed to train_multiple() "
            "must contain at least one complete "
            "transition."
        )

        # --------------------------------------------------------
        # Agent-count assertions
        # --------------------------------------------------------

        if scenario.self_play:
            expected_agent_counts = sorted(
                [
                    scenario.number_of_agents,
                    scenario.number_of_enemies,
                ]
            )
        else:
            expected_agent_counts = [
                scenario.number_of_agents
            ]

        actual_agent_counts = sorted(
            training_call[
                "buffer_agent_counts"
            ]
        )

        assert (
            actual_agent_counts
            == expected_agent_counts
        ), (
            f"Scenario {scenario.name}: expected "
            f"buffer agent counts "
            f"{expected_agent_counts}, received "
            f"{actual_agent_counts}."
        )

        # --------------------------------------------------------
        # Checkpoint assertions
        # --------------------------------------------------------

        run_directory = (
            experiment_directory
            / "run"
        )

        latest_checkpoint = (
            run_directory
            / "r_mappo_latest.pt"
        )

        assert latest_checkpoint.is_file(), (
            "Latest checkpoint was not created: "
            f"{latest_checkpoint}"
        )

        episode_checkpoints = sorted(
            run_directory.glob(
                "r_mappo_episode_*.pt"
            )
        )

        assert episode_checkpoints, (
            "No episode checkpoint was created in "
            f"{run_directory}."
        )

        checkpoint = torch.load(
            latest_checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        assert isinstance(
            checkpoint,
            dict,
        )

        assert (
            checkpoint["episode"]
            == 1
        )

        assert (
            checkpoint["action_dim"]
            == scenario_config[
                "action_dimension"
            ]
        )

        assert (
            checkpoint["max_enemy_slots"]
            == max(
                scenario.number_of_agents,
                scenario.number_of_enemies,
            )
        )

        required_checkpoint_states = {
            "actor_state_dict",
            "target_scorer_state_dict",
            "critic_state_dict",
            "actor_optimizer_state_dict",
            "critic_optimizer_state_dict",
        }

        missing_checkpoint_states = (
            required_checkpoint_states
            - checkpoint.keys()
        )

        assert not missing_checkpoint_states, (
            "Checkpoint is missing states: "
            f"{sorted(missing_checkpoint_states)}"
        )

    finally:
        # training.main() also closes the EnvironmentPool
        # in its finally block. This cleanup only removes
        # files written by the integration scenario.
        if experiment_directory.exists():
            shutil.rmtree(
                experiment_directory
            )