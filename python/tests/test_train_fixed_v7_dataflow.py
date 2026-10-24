
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
OWN_UNIT_IDS = {0, 1, 2}


def _bar_observation_dict(enemy_count=2):
    """
    Create a mock 3vN match.

    There are always three controlled units and between zero and three
    enemies. This lets the test exercise missing enemy slots.
    """
    if not 0 <= enemy_count <= 3:
        raise ValueError("enemy_count must be between 0 and 3")

    formations = [
        # unit_id, team_id, pos_x, pos_y, pos_z, health
        (0, 0, 130.0, 210.0, 0.0, 100.0),
        (1, 0, 170.0, 245.0, 0.0, 82.0),
        (2, 0, 205.0, 190.0, 0.0, 64.0),
        (3, 1, 610.0, 540.0, 0.0, 92.0),
        (4, 1, 665.0, 505.0, 0.0, 74.0),
        (5, 1, 700.0, 570.0, 0.0, 56.0),
    ]

    # Simulate units that are no longer present in the dictionary.
    formations = formations[:3] + formations[3:3 + enemy_count]

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

    # Fallback for enums with a differently represented member.
    action_name = getattr(action_id, "name", str(action_id))
    return "attack" in action_name.lower()


def test_train_fixed_v7_dataflow_one_mappo_update():
    torch.set_num_threads(1)

    env_instances = []
    trainers = []

    # Two enemies deliberately exercise the missing third target slot.
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
                    "own_alive_count": 3,
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
                        "3",
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

        assert len(env.observations) == 3 + enemy_count
        assert len(env.received_actions) == 3
        assert env.step_calls >= 1

        assert len(trainers) == 1
        trainer = trainers[0]

        assert trainer.train_calls == 1
        assert trainer.ppo_epoch == 1

        assert all(
            isinstance(action.action_id, bar_ai.ActionId)
            for action in env.received_actions
        )

        assert all(
            int(action.unit_id) in OWN_UNIT_IDS
            for action in env.received_actions
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

                # An attack may only target a present enemy candidate.
                assert target_id in expected_enemy_ids, (
                    f"Attack targeted unit {target_id}, but valid enemies "
                    f"are {sorted(expected_enemy_ids)}."
                )
            else:
                movement_count += 1

                # Every non-attack action must use the movement target ID.
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

        train_info = trainer.train_info

        # ------------------------------------------------------------
        # Report
        # ------------------------------------------------------------
        raw_actions = [
            {
                "unit_id": int(action.unit_id),
                "team_id": int(action.team_id),
                "ally_team_id": int(action.ally_team_id),
                "action_id": action.action_id,
                "target_unit_id": int(action.target_unit_id),
            }
            for action in env.received_actions
        ]

        print("=" * 60)
        print("DATAFLOW INTEGRATION TEST")
        print("=" * 60)

        print("\n[ENV]")
        print("  simulated BAR environment created")
        print(f"  units in dictionary={len(env.observations)}")
        print(f"  own units={len(OWN_UNIT_IDS)}")
        print(f"  enemy units={enemy_count}")

        print("\n[ROLLOUT]")
        print(f"  steps={env.step_calls}")
        print(f"  actions={len(env.received_actions)}")

        print("\n[TRAIN]")
        print(f"  train calls={trainer.train_calls}")
        print(f"  PPO epochs={trainer.ppo_epoch}")
        print(
            f"  policy_loss="
            f"{float(train_info.get('policy_loss', 0)):.6f}"
        )
        print(
            f"  value_loss="
            f"{float(train_info.get('value_loss', 0)):.6f}"
        )
        print(
            f"  entropy="
            f"{float(train_info.get('dist_entropy', 0)):.6f}"
        )

        print("\n[ACTIONS]")
        for action in raw_actions:
            print(f"  {action}")

        print("\n[TARGET VALIDATION]")
        print(f"  attack actions={attack_count}")
        print(f"  movement actions={movement_count}")
        print(f"  valid enemy IDs={sorted(expected_enemy_ids)}")
        print(f"  movement target ID={MOVEMENT_TARGET_ID}")
        print("\n[CHECKPOINT]")
        print(f"  saved {checkpoint_files[0].name}")

        print("\nRESULT: PASS")
        print("=" * 60, flush=True)


if __name__ == "__main__":
    test_train_fixed_v7_dataflow_one_mappo_update()
