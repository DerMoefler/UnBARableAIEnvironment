from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import bar_ai
from src.train import train_fixed_v7


MOVEMENT_TARGET_ID = 4211

# Unit 0 existiert in diesem Testszenario nicht mehr.
OWN_UNIT_IDS = {1, 2}


def _bar_observation_dict(enemy_count=2):
    """
    Create a simulated match with two controlled units.

    Unit 0 is absent. Only units 1 and 2 are controlled.
    Between zero and three enemies can be present.
    """
    if not 0 <= enemy_count <= 3:
        raise ValueError("enemy_count must be between 0 and 3")

    formations = [
        # unit_id, team_id, pos_x, pos_y, pos_z, health
        (1, 0, 170.0, 245.0, 0.0, 82.0),
        (2, 0, 205.0, 190.0, 0.0, 64.0),
        (3, 1, 610.0, 540.0, 0.0, 92.0),
        (4, 1, 665.0, 505.0, 0.0, 74.0),
        (5, 1, 700.0, 570.0, 0.0, 56.0),
    ]

    # Keep the two own units and add the requested enemy units.
    formations = formations[:2] + formations[2:2 + enemy_count]

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


def _is_attack_action(action):
    """Identify the attack action from the BAR enum."""
    action_id = action.action_id

    attack_enum = getattr(bar_ai.ActionId, "ATTACK", None)
    if attack_enum is not None:
        return action_id == attack_enum

    action_name = getattr(action_id, "name", str(action_id))
    return "attack" in action_name.lower()


def test_train_fixed_v7_dataflow_one_mappo_update():
    torch.set_num_threads(1)

    env_instances = []
    trainers = []

    enemy_count = 2
    expected_enemy_ids = {3, 4}

    class SimulatedBAREnvironment:
        def __init__(self):
            self.observations = _bar_observation_dict(
                enemy_count=enemy_count
            )
            self.received_actions = []
            self.step_calls = 0
            env_instances.append(self)

        def reset(self):
            return self.observations, {"simulated": True}

        def step(self, actions):
            assert isinstance(actions, list)
            assert all(
                isinstance(action, bar_ai.Action)
                for action in actions
            )

            self.step_calls += 1
            self.received_actions.extend(actions)

            return (
                self.observations,
                0.1,
                False,
                False,
                {
                    "own_alive_count": len(OWN_UNIT_IDS),
                    "enemy_alive_count": enemy_count,
                },
            )

        def close(self):
            pass

    class SingleEpochTrainerArgs(train_fixed_v7.TrainerArgs):
        def __init__(self):
            super().__init__()
            self.ppo_epoch = 1

    class CountingRMAPPO(train_fixed_v7.R_MAPPO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.train_calls = 0
            self.train_info = None
            trainers.append(self)

        def train(self, buffer, update_actor=True):
            self.train_calls += 1
            self.train_info = super().train(
                buffer,
                update_actor=update_actor,
            )
            return self.train_info

    with TemporaryDirectory() as model_root:
        with ExitStack() as patches:
            patches.enter_context(
                patch.object(
                    train_fixed_v7,
                    "BAR_Environment",
                    SimulatedBAREnvironment,
                )
            )
            patches.enter_context(
                patch.object(
                    train_fixed_v7,
                    "TrainerArgs",
                    SingleEpochTrainerArgs,
                )
            )
            patches.enter_context(
                patch.object(
                    train_fixed_v7,
                    "R_MAPPO",
                    CountingRMAPPO,
                )
            )
            patches.enter_context(
                patch(
                    "sys.argv",
                    [
                        "train_fixed_v7.py",
                        "--num-episodes",
                        "1",
                        "--num-agents",
                        "2",
                        "--buffer-size",
                        "1",
                        "--num-mini-batch",
                        "1",
                        "--device",
                        "cpu",
                        "--model-root",
                        model_root,
                        "--run-name",
                        "dataflow_test",
                        "--save-every",
                        "0",
                        "--debug-shapes",
                        "--seed",
                        "7",
                    ],
                )
            )

            with redirect_stdout(StringIO()):
                train_fixed_v7.main()

        # ------------------------------------------------------------
        # Training/dataflow assertions
        # ------------------------------------------------------------
        assert len(env_instances) == 1

        env = env_instances[0]

        assert len(env.observations) == len(OWN_UNIT_IDS) + enemy_count
        assert set(
            int(np.asarray(obs).item().unit_id)
            for obs in env.observations.values()
        ) == OWN_UNIT_IDS | expected_enemy_ids

        assert env.step_calls >= 1

        # Exactly two actions are expected: one for unit 1 and one for unit 2.
        assert len(env.received_actions) == len(OWN_UNIT_IDS)

        assert len(trainers) == 1
        trainer = trainers[0]

        assert trainer.train_calls == 1
        assert trainer.ppo_epoch == 1

        assert all(
            isinstance(action.action_id, bar_ai.ActionId)
            for action in env.received_actions
        )

        action_unit_ids = [
            int(action.unit_id)
            for action in env.received_actions
        ]

        # Unit 0 must never receive an action.
        assert 0 not in action_unit_ids, (
            f"Unit 0 unexpectedly received an action: {action_unit_ids}"
        )

        assert set(action_unit_ids) == OWN_UNIT_IDS
        assert len(action_unit_ids) == len(set(action_unit_ids)), (
            "A controlled unit received more than one action in the rollout."
        )

        # ------------------------------------------------------------
        # Target ID assertions
        # ------------------------------------------------------------
        attack_count = 0
        movement_count = 0

        for action in env.received_actions:
            target_id = int(action.target_unit_id)

            if _is_attack_action(action):
                attack_count += 1

                assert target_id in expected_enemy_ids, (
                    f"Attack targeted unit {target_id}, but valid enemies "
                    f"are {sorted(expected_enemy_ids)}."
                )
            else:
                movement_count += 1

                assert target_id == MOVEMENT_TARGET_ID, (
                    f"Movement action used target {target_id}; "
                    f"expected {MOVEMENT_TARGET_ID}."
                )

        # ------------------------------------------------------------
        # Checkpoint assertions
        # ------------------------------------------------------------
        run_dir = Path(model_root) / "dataflow_test"

        assert (run_dir / "r_mappo_latest.pt").exists()
        assert not (run_dir / "native.log").exists()

        checkpoint_files = list(
            run_dir.glob("r_mappo_episode_*.pt")
        )
        assert checkpoint_files

        train_info = trainer.train_info or {}

        # ------------------------------------------------------------
        # Report
        # ------------------------------------------------------------
        separator = "=" * 76
        line = "-" * 76

        print(f"\n\n{separator}")
        print("  MAPPO INTEGRATION TEST — DATAFLOW REPORT")
        print(separator)

        print("\n[1] ENVIRONMENT")
        print(line)
        print(f"  Environment       : {type(env).__name__}")
        print(f"  Total units       : {len(env.observations)}")
        print(f"  Controlled units  : {sorted(OWN_UNIT_IDS)}")
        print(f"  Removed unit      : 0")
        print(f"  Enemy units       : {sorted(expected_enemy_ids)}")
        print(f"  Environment steps : {env.step_calls}")

        print("\n[2] TRAINING")
        print(line)
        print(f"  Trainer           : {type(trainer).__name__}")
        print(f"  Training calls    : {trainer.train_calls}")
        print(f"  PPO epochs        : {trainer.ppo_epoch}")

        for key in ("policy_loss", "value_loss", "dist_entropy"):
            value = train_info.get(key)
            if value is not None:
                print(f"  {key:<20}: {value}")

        print("\n[3] ACTION OUTPUT")
        print(line)
        print(f"  Total actions     : {len(env.received_actions)}")
        print(f"  Attack actions    : {attack_count}")
        print(f"  Movement actions  : {movement_count}")

        print("\n  Actions received by the environment:")

        for index, action in enumerate(env.received_actions):
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
                f"target={int(action.target_unit_id):>4}"
            )

        print("\n[4] VALIDATION")
        print(line)
        print(f"  Active own IDs    : {sorted(OWN_UNIT_IDS)}")
        print(f"  Unit 0 has action : {0 in action_unit_ids}")
        print(f"  Valid enemy IDs   : {sorted(expected_enemy_ids)}")
        print(f"  Movement target   : {MOVEMENT_TARGET_ID}")
        print("  Unit validation   : PASS")
        print("  Target validation : PASS")

        print("\n[5] CHECKPOINTS")
        print(line)
        print(f"  Latest checkpoint : {run_dir / 'r_mappo_latest.pt'}")
        print(f"  Episode checkpoint: {checkpoint_files[0].name}")
        print(
            f"  Native log absent : "
            f"{not (run_dir / 'native.log').exists()}"
        )

        print(f"\n{separator}")
        print("  RESULT: PASS")
        print(f"{separator}\n", flush=True)
