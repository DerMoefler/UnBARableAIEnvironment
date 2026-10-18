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


def _bar_observation_dict():
    formations = (
        (0, 0, 130.0, 210.0, 0.0, 100.0),
        (1, 0, 170.0, 245.0, 0.0, 82.0),
        (2, 0, 205.0, 190.0, 0.0, 64.0),
        (3, 1, 610.0, 540.0, 0.0, 92.0),
        (4, 1, 665.0, 505.0, 0.0, 74.0),
        (5, 1, 700.0, 570.0, 0.0, 56.0),
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
        for unit_id, team_id, pos_x, pos_z, pos_y, health in formations
    }


def test_train_fixed_v7_dataflow_one_mappo_update():
    torch.set_num_threads(1)
    env_instances = []
    trainers = []

    class SimulatedBAREnvironment:
        def __init__(self):
            self.observations = _bar_observation_dict()
            self.received_actions = []
            self.step_calls = 0
            env_instances.append(self)

        def reset(self):
            return self.observations, {"simulated": True}

        def step(self, actions):
            assert isinstance(actions, list)
            assert all(isinstance(action, bar_ai.Action) for action in actions)
            self.step_calls += 1
            self.received_actions.extend(actions)
            return (
                self.observations,
                0.1,
                False,
                False,
                {"own_alive_count": 3, "enemy_alive_count": 3},
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
            self.train_info = super().train(buffer, update_actor=update_actor)
            return self.train_info

    with TemporaryDirectory() as model_root:
        with ExitStack() as patches:
            patches.enter_context(
                patch.object(train_fixed_v7, "BAR_Environment", SimulatedBAREnvironment)
            )
            patches.enter_context(
                patch.object(train_fixed_v7, "TrainerArgs", SingleEpochTrainerArgs)
            )
            patches.enter_context(
                patch.object(train_fixed_v7, "R_MAPPO", CountingRMAPPO)
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

        assert len(env_instances) == 1
        assert len(env_instances[0].received_actions) == 3
        raw_actions = [
            {
                "unit_id": action.unit_id,
                "team_id": action.team_id,
                "ally_team_id": action.ally_team_id,
                "action_id": action.action_id,
                "target_unit_id": action.target_unit_id,
            }
            for action in env_instances[0].received_actions
        ]
        assert trainers[0].train_calls == 1
        assert trainers[0].ppo_epoch == 1
        assert all(
            isinstance(action.action_id, bar_ai.ActionId)
            for action in env_instances[0].received_actions
        )
        assert all(
            int(action.unit_id) in {0, 1, 2}
            for action in env_instances[0].received_actions
        )
        assert (
            Path(model_root) / "dataflow_test" / "r_mappo_latest.pt"
        ).exists()
        assert not (
            Path(model_root) / "dataflow_test" / "native.log"
        ).exists()

        train_info = trainers[0].train_info
        action_ids = [int(action.action_id) for action in env_instances[0].received_actions]
        unit_ids = [int(action.unit_id) for action in env_instances[0].received_actions]
        checkpoint_name = next(
            path.name
            for path in (Path(model_root) / "dataflow_test").glob("r_mappo_episode_*.pt")
        )

        print("=" * 60)
        print("DATAFLOW INTEGRATION TEST")
        print("=" * 60)
        print("\n[test] launch train_fixed_v7\n")
        print("[MAPPO]")
        print("  initialization complete")
        print("  agents=3")
        print("  device=cpu\n")
        print("[ENV]")
        print("  simulated BAR environment created")
        print(f"  reset returned {len(env_instances[0].observations)} units\n")
        print("[ROLLOUT]")
        print(f"  step={env_instances[0].step_calls}")
        print(f"  actions={len(env_instances[0].received_actions)}\n")
        print("[TRAIN]")
        print("  MAPPO update started")
        print(f"  policy_loss={float(train_info.get('policy_loss', 0)):.6f}")
        print(f"  value_loss={float(train_info.get('value_loss', 0)):.6f}")
        print(f"  entropy={float(train_info.get('dist_entropy', 0)):.6f}\n")
        print("[ACTIONS]")
        for action in raw_actions:
            print(f"  {action}")
        print()
        print("[CHECKPOINT]")
        print(f"  saved {checkpoint_name}\n")
        print("RESULT: PASS")
        print("=" * 60, flush=True)


if __name__ == "__main__":
    test_train_fixed_v7_dataflow_one_mappo_update()