from __future__ import annotations

import shutil
import uuid

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from src.train import training
from tests.src.helpers import (
    create_scenario_config,
    load_base_config,
    write_config,
    IntegrationScenario,
)


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

    base_config = load_base_config()

    scenario_config = (
        create_scenario_config(
            base_config=base_config,
            scenario=scenario,
            run_name=run_name,
        )
    )

    write_config(
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

        model_name = scenario_config[
            "model_name"
        ]

        configured_model_path = Path(
            model_name
        )

        latest_checkpoint = (
            run_directory
            / model_name
        )

        assert latest_checkpoint.is_file(), (
            "Latest checkpoint was not created: "
            f"{latest_checkpoint}"
        )

        episode_checkpoints = sorted(
            run_directory.glob(
                f"{configured_model_path.stem}"
                "_episode_*"
                f"{configured_model_path.suffix}"
            )
        )

        assert episode_checkpoints, (
            "No episode checkpoint matching "
            f"{configured_model_path.stem}"
            "_episode_*"
            f"{configured_model_path.suffix} "
            f"was created in {run_directory}. "
            f"Existing files: "
            f"{sorted(path.name for path in run_directory.iterdir())}"
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