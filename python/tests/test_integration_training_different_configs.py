from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import bar_ai
from src.train import training
from src.train.mappo import trainerargs


MOVEMENT_TARGET_ID = 4211


@dataclass(frozen=True)
class TrainingScenario:
    name: str
    self_play: bool
    number_of_agents: int
    number_of_enemies: int
    expected_train_calls: int

    @property
    def team_ids(self) -> tuple[int, ...]:
        if self.self_play:
            return (0, 1)

        return (0,)


SCENARIOS = (
    TrainingScenario(
        name="normal_3v3",
        self_play=False,
        number_of_agents=3,
        number_of_enemies=3,
        expected_train_calls=1,
    ),
    TrainingScenario(
        name="selfplay_3v3",
        self_play=True,
        number_of_agents=3,
        number_of_enemies=3,
        expected_train_calls=2,
    ),
    TrainingScenario(
        name="normal_2v2",
        self_play=False,
        number_of_agents=2,
        number_of_enemies=2,
        expected_train_calls=1,
    ),
        TrainingScenario(
        name="selfplay_3v5",
        self_play=True,
        number_of_agents=3,
        number_of_enemies=5,
        expected_train_calls=2,
    ),
)


def _make_unit(
    *,
    unit_id: int,
    team_id: int,
    pos_x: float,
    pos_z: float,
) -> np.ndarray:
    return np.asarray(
        SimpleNamespace(
            unit_id=unit_id,
            team_id=team_id,
            ally_team_id=team_id,
            unit_def_id=102,
            health=100.0,
            max_health=100.0,
            pos_x=pos_x,
            pos_y=0.0,
            pos_z=pos_z,
            los_radius=350.0,
            is_dead=False,
            being_built=False,
        )
    )


def _bar_observation_dict(
    *,
    number_of_agents: int,
    number_of_enemies: int,
) -> dict[str, np.ndarray]:
    observations: dict[str, np.ndarray] = {}

    for index in range(number_of_agents):
        unit_id = index + 1

        observations[f"unit_{unit_id}"] = _make_unit(
            unit_id=unit_id,
            team_id=0,
            pos_x=100.0 + index * 50.0,
            pos_z=200.0,
        )

    for index in range(number_of_enemies):
        unit_id = number_of_agents + index + 1

        observations[f"unit_{unit_id}"] = _make_unit(
            unit_id=unit_id,
            team_id=1,
            pos_x=600.0 + index * 50.0,
            pos_z=500.0,
        )

    return observations


def _controlled_unit_ids(
    *,
    team_id: int,
    number_of_agents: int,
    number_of_enemies: int,
) -> set[int]:
    if team_id == 0:
        return set(
            range(
                1,
                number_of_agents + 1,
            )
        )

    return set(
        range(
            number_of_agents + 1,
            number_of_agents
            + number_of_enemies
            + 1,
        )
    )


def _enemy_unit_ids(
    *,
    team_id: int,
    number_of_agents: int,
    number_of_enemies: int,
) -> set[int]:
    return _controlled_unit_ids(
        team_id=1 - team_id,
        number_of_agents=number_of_agents,
        number_of_enemies=number_of_enemies,
    )


def _is_attack_action(
    action: bar_ai.Action,
) -> bool:
    action_id = action.action_id

    attack_enum = getattr(
        bar_ai.ActionId,
        "ATTACK",
        None,
    )

    if attack_enum is not None:
        return action_id == attack_enum

    action_name = getattr(
        action_id,
        "name",
        str(action_id),
    )

    return "attack" in action_name.lower()


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=lambda scenario: scenario.name,
)
def test_training_dataflow(
    scenario: TrainingScenario,
    tmp_path: Path,
):
    torch.set_num_threads(1)

    env_instances = []
    trainers = []

    class SimulatedBAREnvironment:
        def __init__(
            self,
            engine_id=0,
            port=50051,
        ):
            self.unbarable_ai_team_ids = (
                scenario.team_ids
            )

            self.current_team_index = 0

            self.current_team_id = (
                self.unbarable_ai_team_ids[0]
            )

            self.observations = (
                _bar_observation_dict(
                    number_of_agents=(
                        scenario.number_of_agents
                    ),
                    number_of_enemies=(
                        scenario.number_of_enemies
                    ),
                )
            )

            self.received_actions: list[
                bar_ai.Action
            ] = []

            self.actions_per_step: list[
                tuple[int, list[bar_ai.Action]]
            ] = []

            self.step_calls = 0

            env_instances.append(self)

        def reset(self):
            self.current_team_index = 0

            self.current_team_id = (
                self.unbarable_ai_team_ids[0]
            )

            self.step_calls = 0
            self.received_actions.clear()
            self.actions_per_step.clear()

            return (
                self.observations,
                {
                    "simulated": True,
                    "current_team_id": (
                        self.current_team_id
                    ),
                    "acting_team_id": None,
                    "reward_team_id": None,
                    "episode_step": 0,
                    "max_episode_steps": (
                        len(
                            self.unbarable_ai_team_ids
                        )
                    ),
                    "reward_state_initialized": True,
                },
            )

        def step(
            self,
            actions: list[bar_ai.Action],
        ):
            acting_team_id = (
                self.current_team_id
            )

            assert isinstance(actions, list)

            assert all(
                isinstance(
                    action,
                    bar_ai.Action,
                )
                for action in actions
            )

            expected_unit_ids = (
                _controlled_unit_ids(
                    team_id=acting_team_id,
                    number_of_agents=(
                        scenario.number_of_agents
                    ),
                    number_of_enemies=(
                        scenario.number_of_enemies
                    ),
                )
            )

            received_unit_ids = {
                int(action.unit_id)
                for action in actions
            }

            assert (
                received_unit_ids
                == expected_unit_ids
            ), (
                f"Team {acting_team_id}: "
                f"expected actions for units "
                f"{sorted(expected_unit_ids)}, "
                f"received "
                f"{sorted(received_unit_ids)}"
            )

            received_team_ids = {
                int(action.team_id)
                for action in actions
            }

            assert received_team_ids == {
                acting_team_id
            }

            self.step_calls += 1

            self.actions_per_step.append(
                (
                    acting_team_id,
                    list(actions),
                )
            )

            self.received_actions.extend(
                actions
            )

            self.current_team_index = (
                self.current_team_index + 1
            ) % len(
                self.unbarable_ai_team_ids
            )

            self.current_team_id = (
                self.unbarable_ai_team_ids[
                    self.current_team_index
                ]
            )

            # Jeder UnBARableAI-Player soll in der Episode
            # genau einmal handeln.
            terminated = (
                self.step_calls
                >= len(
                    self.unbarable_ai_team_ids
                )
            )

            return (
                self.observations,
                0.1,
ated,
                False,
                {
                    "acting_team_id": (
                        acting_team_id
                    ),
                    "current_team_id": (
                        self.current_team_id
                    ),
                    "reward_team_id": (
                        acting_team_id
                    ),
                    "episode_step": (
                        self.step_calls
                    ),
                    "own_alive_count": len(
                        expected_unit_ids
                    ),
                    "enemy_alive_count": len(
                        _enemy_unit_ids(
                            team_id=acting_team_id,
                            number_of_agents=(
                                scenario.number_of_agents
                            ),
                            number_of_enemies=(
                                scenario.number_of_enemies
                            ),
                        )
                    ),
                    "terminated": terminated,
                    "truncated": False,
                    "episode_outcome": (
                        "won"
                        if terminated
                        else None
                    ),
                    "reward": 0.1,
                    "reward_info": {
                        "enemy_damage_done": 0.0,
                        "own_damage_taken": 0.0,
                    },
                },
            )

        def close(self):
            pass

    class SimulatedEnvironmentPool:
        def __init__(
            self,
            config_path,
        ):
            self.config_path = config_path
            self.environments = []

        def initialize_environements(self):
            self.environments = [
                SimulatedBAREnvironment()
            ]

        def close_environements(self):
            for environment in (
                self.environments
            ):
                environment.close()

            self.environments.clear()

    class SingleEpochTrainerArgs(
        trainerargs.TrainerArgs
    ):
        def __init__(self):
            super().__init__()
            self.ppo_epoch = 1

    class CountingRMAPPO(
        training.R_MAPPO
    ):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(
                *args,
                **kwargs,
            )

            self.train_calls = 0
            self.train_infos = []

            trainers.append(self)

        def train(
            self,
            buffer,
            update_actor=True,
        ):
            self.train_calls += 1

            train_info = super().train(
                buffer,
                update_actor=update_actor,
            )

            self.train_infos.append(
                train_info
            )

            return train_info

    config_path = Path(
        "src/train/experiments/3v3Test/"
        "3v3Test.json"
    )

    (
        config,
        training_config,
        mappo_config,
    ) = training.load_training_config(
        config_path
    )

    config = config.copy()
    training_config = (
        training_config.copy()
    )
    mappo_config = mappo_config.copy()

    run_name = (
        f"dataflow_test_{scenario.name}"
    )

    config.update(
        {
            "run_name": run_name,
            "device": "cpu",
            "model_name": (
                "r_mappo_latest.pt"
            ),
            "number_of_agents": (
                scenario.number_of_agents
            ),
            "number_of_enemies": (
                scenario.number_of_enemies
            ),
            "action_dimension": 5,
            "self_play": scenario.self_play,
            "adapter": {
                "position_scale": 1000.0,
            },
        }
    )

    training_config.update(
        {
            "continue_training": False,

            # Pro Team wird genau ein Übergang
            # gespeichert.
            "buffer_size": 1,

            "mini_batch": 1,
            "episodes": 1,
            "number_of_environments": 1,
            "number_of_engines": 1,
            "save_every_x_episode": 0,

            # Das Mock terminiert bereits selbst.
            "maximum_steps": max(
                1,
                len(scenario.team_ids),
            ),
        }
    )

    mappo_config.update(
        {
            "learning_rate": 1e-4,
            "gamma": 0.99,
        }
    )

    training_config["mappo"] = (
        mappo_config
    )

    config["training"] = (
        training_config
    )

    with ExitStack() as patches:
        patches.enter_context(
            patch.object(
                training,
                "EnvironmentPool",
                SimulatedEnvironmentPool,
            )
        )

        patches.enter_context(
            patch.object(
                training,
                "TrainerArgs",
                SingleEpochTrainerArgs,
            )
        )

        patches.enter_context(
            patch.object(
                training,
                "R_MAPPO",
                CountingRMAPPO,
            )
        )

        patches.enter_context(
            patch.object(
                training,
                "load_training_config",
                return_value=(
                    config,
                    training_config,
                    mappo_config,
                ),
            )
        )

        # Verhindert, dass der Test Checkpoints in
        # das echte Experimentverzeichnis schreibt.
        original_save_checkpoint = (
            training._save_checkpoint
        )

        test_run_directory = (
            tmp_path
            / run_name
            / "run"
        )

        def save_checkpoint_to_tmp(
            policy,
            adapter,
            episode,
            model_dir,
        ):
            return original_save_checkpoint(
                policy,
                adapter,
                episode,
                test_run_directory,
            )

        patches.enter_context(
            patch.object(
                training,
                "_save_checkpoint",
                save_checkpoint_to_tmp,
            )
        )

        training.main()

    # ------------------------------------------------------------
    # Environment assertions
    # ------------------------------------------------------------

    assert len(env_instances) == 1

    environment = env_instances[0]

    assert (
        environment.unbarable_ai_team_ids
        == scenario.team_ids
    )

    assert environment.step_calls == len(
        scenario.team_ids
    )

    expected_total_actions = sum(
        len(
            _controlled_unit_ids(
                team_id=team_id,
                number_of_agents=(
                    scenario.number_of_agents
                ),
                number_of_enemies=(
                    scenario.number_of_enemies
                ),
            )
        )
        for team_id in scenario.team_ids
    )

    assert (
        len(environment.received_actions)
        == expected_total_actions
    )

    for (
        acting_team_id,
        actions,
    ) in environment.actions_per_step:
        expected_unit_ids = (
            _controlled_unit_ids(
                team_id=acting_team_id,
                number_of_agents=(
                    scenario.number_of_agents
                ),
                number_of_enemies=(
                    scenario.number_of_enemies
                ),
            )
        )

        action_unit_ids = {
            int(action.unit_id)
            for action in actions
        }

        action_team_ids = {
            int(action.team_id)
            for action in actions
        }

        assert (
            action_unit_ids
            == expected_unit_ids
        )

        assert action_team_ids == {
            acting_team_id
        }

        assert len(actions) == len(
            expected_unit_ids
        )

    # ------------------------------------------------------------
    # Training assertions
    # ------------------------------------------------------------

    assert len(trainers) == 1

    trainer = trainers[0]

    assert (
        trainer.train_calls
        == scenario.expected_train_calls
    )

    assert (
        len(trainer.train_infos)
        == scenario.expected_train_calls
    )

    assert trainer.ppo_epoch == 1

    # ------------------------------------------------------------
    # Action target assertions
    # ------------------------------------------------------------

    for (
        acting_team_id,
        actions,
    ) in environment.actions_per_step:
        expected_enemy_ids = (
            _enemy_unit_ids(
                team_id=acting_team_id,
                number_of_agents=(
                    scenario.number_of_agents
                ),
                number_of_enemies=(
                    scenario.number_of_enemies
                ),
            )
        )

        for action in actions:
            target_id = int(
                action.target_unit_id
            )

            if _is_attack_action(action):
                assert (
                    target_id
                    in expected_enemy_ids
                )
            else:
                assert (
                    target_id
                    == MOVEMENT_TARGET_ID
                )

    # ------------------------------------------------------------
    # Checkpoint assertions
    # ------------------------------------------------------------

    assert (
        test_run_directory
        / "r_mappo_latest.pt"
    ).exists()

    checkpoint_files = list(
        test_run_directory.glob(
            "r_mappo_episode_*.pt"
        )
    )

    assert checkpoint_files