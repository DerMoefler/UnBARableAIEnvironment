from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import bar_ai
from src.train import training
from src.train.mappo import trainerargs


MOVEMENT_TARGET_ID = 4211

# Unit 0 existiert in diesem Testszenario nicht mehr.
OWN_UNIT_IDS = {1, 2, 3}


def test_create_mappo_adapters_uses_training_dimensions() -> None:
    config = {
        "adapter": {"position_scale": 1000.0},
        "action_dimension": 5,
        "number_of_agents": 3,
        "number_of_enemies": 5,
    }

    adapters = training.create_mappo_adapters(
        2,
        config,
        team_id=1,
    )

    assert len(adapters) == 2
    assert all(adapter.num_agents == 3 for adapter in adapters)
    assert all(adapter.training_team_id == 1 for adapter in adapters)
    assert all(adapter.max_enemy_slots == 5 for adapter in adapters)


def _bar_observation_dict(
    enemy_count: int = 3,
) -> dict[str, np.ndarray]:
    """
    Create a simulated match with three controlled units.

    Unit 0 is absent. Only units 1, 2 and 3 are controlled.
    Between zero and three enemies can be present.
    """
    if not 0 <= enemy_count <= 3:
        raise ValueError(
            "enemy_count must be between 0 and 3"
        )

    formations = [
        # unit_id, team_id, pos_x, pos_y, pos_z, health

        # Eigene Einheiten, Team 0
        (1, 0, 170.0, 245.0, 0.0, 82.0),
        (2, 0, 205.0, 190.0, 0.0, 64.0),
        (3, 0, 240.0, 220.0, 0.0, 91.0),

        # Gegnerische Einheiten, Team 1
        (4, 1, 610.0, 540.0, 0.0, 92.0),
        (5, 1, 665.0, 505.0, 0.0, 74.0),
        (6, 1, 700.0, 570.0, 0.0, 56.0),
    ]

    formations = (
        formations[:3]
        + formations[
            3:3 + enemy_count
        ]
    )

    return {
        f"agent_{unit_id}": np.asarray(
            SimpleNamespace(
                unit_id=unit_id,
                team_id=team_id,
                ally_team_id=team_id,
                unit_def_id=102,
                health=health,
                max_health=100.0,
                pos_x=pos_x,
                pos_y=pos_y,
                pos_z=pos_z,
                los_radius=350.0,
                is_dead=False,
                being_built=False,
            )
        )
        for (
            unit_id,
            team_id,
            pos_x,
            pos_y,
            pos_z,
            health,
        ) in formations
    }


def _is_attack_action(
    action: bar_ai.Action,
) -> bool:
    """
    Identify the attack action from the BAR enum.
    """
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

    return (
        "attack"
        in action_name.lower()
    )


def test_training_dataflow_one_combined_mappo_update(
    tmp_path: Path,
):
    torch.set_num_threads(1)

    env_instances = []
    trainers = []

    enemy_count = 3
    expected_enemy_ids = {4, 5, 6}

    class SimulatedBAREnvironment:
        def __init__(
            self,
            engine_id=0,
            port=50051,
        ):
            self.unbarable_ai_team_ids = (
                0,
            )

            self.current_team_id = (
                self.unbarable_ai_team_ids[0]
            )

            self.observations = (
                _bar_observation_dict(
                    enemy_count=enemy_count
                )
            )

            self.received_actions: list[
                bar_ai.Action
            ] = []

            self.actions_per_step: list[
                list[bar_ai.Action]
            ] = []

            self.step_calls = 0

            env_instances.append(self)

        def reset(self):
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
                    "max_episode_steps": 1,
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

            assert isinstance(
                actions,
                list,
            )

            assert all(
                isinstance(
                    action,
                    bar_ai.Action,
                )
                for action in actions
            )

            assert all(
                int(action.team_id)
                == acting_team_id
                for action in actions
            )

            self.step_calls += 1

            self.actions_per_step.append(
                list(actions)
            )

            self.received_actions.extend(
                actions
            )

            terminated = (
                self.step_calls >= 1
            )

            # Bei nur einer UnBARableAI ist der nächste
            # Player erneut Team 0. Reward und neue
            # Observation gehören damit ebenfalls Team 0.
            self.current_team_id = (
                self.unbarable_ai_team_ids[0]
            )

            return (
                self.observations,
                0.1,
                terminated,
                False,
                {
                    "acting_team_id": (
                        acting_team_id
                    ),
                    "current_team_id": (
                        self.current_team_id
                    ),
                    "reward_team_id": (
                        self.current_team_id
                    ),
                    "episode_step": (
                        self.step_calls
                    ),
                    "own_alive_count": len(
                        OWN_UNIT_IDS
                    ),
                    "enemy_alive_count": (
                        enemy_count
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
            self.train_multiple_calls = 0
            self.train_info = None
            self.trained_buffer_count = None
            self.trained_buffer_steps = None
            self.trained_buffer_agent_counts = None

            trainers.append(self)

        def train(
            self,
            buffer,
            update_actor=True,
        ):
            self.train_calls += 1

            self.train_info = super().train(
                buffer,
                update_actor=update_actor,
            )

            return self.train_info

        def train_multiple(
            self,
            buffers,
            update_actor=True,
        ):
            buffer_list = list(
                buffers
            )

            self.train_multiple_calls += 1

            self.trained_buffer_count = len(
                buffer_list
            )

            self.trained_buffer_steps = [
                int(buffer.step)
                for buffer in buffer_list
            ]

            self.trained_buffer_agent_counts = [
                int(buffer.num_agents)
                for buffer in buffer_list
            ]

            self.train_info = (
                super().train_multiple(
                    buffer_list,
                    update_actor=update_actor,
                )
            )

            return self.train_info

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
    mappo_config = (
        mappo_config.copy()
    )

    training_config["mappo"] = (
        mappo_config
    )

    config["training"] = (
        training_config
    )

    run_name = (
        "dataflow_test"
    )

    config.update(
        {
            "run_name": run_name,
            "device": "cpu",
            "model_name": (
                "r_mappo_latest.pt"
            ),
            "number_of_agents": 3,
            "number_of_enemies": 3,
            "action_dimension": 5,
            "self_play": False,
            "adapter": {
                "position_scale": 1000.0,
            },
        }
    )

    training_config.update(
        {
            "continue_training": False,
            "buffer_size": 1,
            "mini_batch": 1,
            "episodes": 1,
            "number_of_environments": 1,
            "number_of_engines": 1,
            "save_every_x_episode": 0,
            "maximum_steps": 1,
        }
    )

    mappo_config.update(
        {
            "learning_rate": 1e-4,
            "gamma": 0.99,
        }
    )

    test_run_directory = (
        tmp_path
        / run_name
        / "run"
    )

    original_save_checkpoint = (
        training._save_checkpoint
    )

    def save_checkpoint_to_tmp(
        policy,
        adapter,
        episode,
        model_dir,
        *,
        model_name,
    ):
        return original_save_checkpoint(
            policy,
            adapter,
            episode,
            test_run_directory,
            model_name=model_name,
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

        patches.enter_context(
            patch.object(
                training,
                "_save_checkpoint",
                save_checkpoint_to_tmp,
            )
        )

        training.main(
            config_path=config_path
        )

    # ------------------------------------------------------------
    # Environment assertions
    # ------------------------------------------------------------

    assert len(env_instances) == 1

    env = env_instances[0]

    assert env.step_calls == 1

    assert (
        env.unbarable_ai_team_ids
        == (0,)
    )

    assert env.current_team_id == 0

    assert (
        len(env.observations)
        == len(OWN_UNIT_IDS)
        + enemy_count
    )

    observed_unit_ids = {
        int(
            np.asarray(
                observation
            ).item().unit_id
        )
        for observation
        in env.observations.values()
    }

    assert observed_unit_ids == (
        OWN_UNIT_IDS
        | expected_enemy_ids
    )

    assert (
        len(env.actions_per_step)
        == env.step_calls
    )

    # ------------------------------------------------------------
    # Action assertions
    # ------------------------------------------------------------

    for step_index, actions in enumerate(
        env.actions_per_step
    ):
        assert len(actions) == len(
            OWN_UNIT_IDS
        ), (
            f"Step {step_index}: expected "
            f"{len(OWN_UNIT_IDS)} actions, "
            f"received {len(actions)}"
        )

        action_unit_ids = [
            int(action.unit_id)
            for action in actions
        ]

        assert set(
            action_unit_ids
        ) == OWN_UNIT_IDS, (
            f"Step {step_index}: unexpected "
            f"unit IDs {action_unit_ids}"
        )

        assert len(
            action_unit_ids
        ) == len(
            set(action_unit_ids)
        ), (
            f"Step {step_index}: duplicate "
            f"unit actions {action_unit_ids}"
        )

        action_team_ids = [
            int(action.team_id)
            for action in actions
        ]

        assert set(
            action_team_ids
        ) == {0}, (
            f"Step {step_index}: expected all "
            f"actions for team 0, received "
            f"{action_team_ids}"
        )

    assert all(
        isinstance(
            action.action_id,
            bar_ai.ActionId,
        )
        for action in env.received_actions
    )

    action_unit_ids = [
        int(action.unit_id)
        for action in env.received_actions
    ]

    # Unit 0 darf keine Action erhalten.
    assert 0 not in action_unit_ids

    assert set(
        action_unit_ids
    ) == OWN_UNIT_IDS

    expected_action_count = (
        env.step_calls
        * len(OWN_UNIT_IDS)
    )

    assert len(
        action_unit_ids
    ) == expected_action_count

    # ------------------------------------------------------------
    # Combined PPO assertions
    # ------------------------------------------------------------

    assert len(trainers) == 1

    trainer = trainers[0]

    # training.py ruft direkt train_multiple() auf.
    assert trainer.train_calls == 0

    assert (
        trainer.train_multiple_calls
        == 1
    )

    # Ohne Self-Play wird genau ein Buffer gemeinsam trainiert.
    assert (
        trainer.trained_buffer_count
        == 1
    )

    assert (
        trainer.trained_buffer_steps
        == [1]
    )

    assert (
        trainer.trained_buffer_agent_counts
        == [len(OWN_UNIT_IDS)]
    )

    assert trainer.ppo_epoch == 1

    assert isinstance(
        trainer.train_info,
        dict,
    )

    train_info = (
        trainer.train_info
        or {}
    )

    for key in (
        "policy_loss",
        "value_loss",
        "dist_entropy",
        "actor_grad_norm",
        "critic_grad_norm",
        "ratio",
    ):
        assert key in train_info

        value = float(
            train_info[key]
        )

        assert np.isfinite(value), (
            f"Training metric {key!r} "
            f"is not finite: {value}"
        )

    # ------------------------------------------------------------
    # Target assertions
    # ------------------------------------------------------------

    attack_count = 0
    movement_count = 0

    for action in env.received_actions:
        target_id = int(
            action.target_unit_id
        )

        if _is_attack_action(action):
            attack_count += 1

            assert (
                target_id
                in expected_enemy_ids
            ), (
                f"Attack targeted unit {target_id}, "
                "but valid enemies are "
                f"{sorted(expected_enemy_ids)}."
            )
        else:
            movement_count += 1

            assert (
                target_id
                == MOVEMENT_TARGET_ID
            ), (
                f"Movement action used target "
                f"{target_id}; expected "
                f"{MOVEMENT_TARGET_ID}."
            )

    # ------------------------------------------------------------
    # Checkpoint assertions
    # ------------------------------------------------------------

    latest_checkpoint = (
        test_run_directory
        / config["model_name"]
    )

    assert latest_checkpoint.exists()

    model_path = Path(
        config["model_name"]
    )

    checkpoint_files = list(
        test_run_directory.glob(
            f"{model_path.stem}"
            "_episode_*"
            f"{model_path.suffix}"
        )
    )

    assert checkpoint_files

    checkpoint = torch.load(
        latest_checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["episode"] == 1
    assert checkpoint["action_dim"] == 5
    assert checkpoint["max_enemy_slots"] == 3

    assert "actor_state_dict" in checkpoint
    assert "target_scorer_state_dict" in checkpoint
    assert "critic_state_dict" in checkpoint

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    separator = "=" * 76
    line = "-" * 76

    print(f"\n\n{separator}")
    print(
        "  MAPPO DATAFLOW TEST REPORT"
    )
    print(separator)

    print("\n[1] ENVIRONMENT")
    print(line)
    print(
        f"  Environment       : "
        f"{type(env).__name__}"
    )
    print(
        f"  Total units       : "
        f"{len(env.observations)}"
    )
    print(
        f"  Controlled units  : "
        f"{sorted(OWN_UNIT_IDS)}"
    )
    print(
        "  Removed unit      : 0"
    )
    print(
        f"  Enemy units       : "
        f"{sorted(expected_enemy_ids)}"
    )
    print(
        f"  Environment steps : "
        f"{env.step_calls}"
    )

    print("\n[2] TRAINING")
    print(line)
    print(
        f"  Trainer                  : "
        f"{type(trainer).__name__}"
    )
    print(
        f"  train() calls            : "
        f"{trainer.train_calls}"
    )
    print(
        f"  train_multiple() calls   : "
        f"{trainer.train_multiple_calls}"
    )
    print(
        f"  Combined buffers         : "
        f"{trainer.trained_buffer_count}"
    )
    print(
        f"  Buffer steps             : "
        f"{trainer.trained_buffer_steps}"
    )
    print(
        f"  Buffer agent counts      : "
        f"{trainer.trained_buffer_agent_counts}"
    )
    print(
        f"  PPO epochs               : "
        f"{trainer.ppo_epoch}"
    )

    for key in (
        "policy_loss",
        "value_loss",
        "dist_entropy",
    ):
        print(
            f"  {key:<24}: "
            f"{train_info[key]}"
        )

    print("\n[3] ACTION OUTPUT")
    print(line)
    print(
        f"  Total actions     : "
        f"{len(env.received_actions)}"
    )
    print(
        f"  Attack actions    : "
        f"{attack_count}"
    )
    print(
        f"  Movement actions  : "
        f"{movement_count}"
    )

    print(
        "\n  Actions received by "
        "the environment:"
    )

    for index, action in enumerate(
        env.received_actions
    ):
        action_name = getattr(
            action.action_id,
            "name",
            str(action.action_id),
        )

        print(
            f"    [{index:02d}] "
            f"unit={int(action.unit_id):>3} | "
            f"team={int(action.team_id):>2} | "
            f"action={action_name:<12} | "
            f"target="
            f"{int(action.target_unit_id):>4}"
        )

    print("\n[4] VALIDATION")
    print(line)
    print(
        f"  Active own IDs    : "
        f"{sorted(OWN_UNIT_IDS)}"
    )
    print(
        f"  Unit 0 has action : "
        f"{0 in action_unit_ids}"
    )
    print(
        f"  Valid enemy IDs   : "
        f"{sorted(expected_enemy_ids)}"
    )
    print(
        f"  Movement target   : "
        f"{MOVEMENT_TARGET_ID}"
    )
    print(
        "  Unit validation   : PASS"
    )
    print(
        "  Target validation : PASS"
    )
    print(
        "  Combined PPO      : PASS"
    )

    print("\n[5] CHECKPOINTS")
    print(line)
    print(
        f"  Latest checkpoint : "
        f"{latest_checkpoint}"
    )
    print(
        f"  Episode checkpoint: "
        f"{checkpoint_files[0].name}"
    )

    print(f"\n{separator}")
    print("  RESULT: PASS")
    print(
        f"{separator}\n",
        flush=True,
    )