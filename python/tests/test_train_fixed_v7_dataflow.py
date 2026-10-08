from contextlib import ExitStack
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
            env_instances.append(self)
            print("[test] Simulated BAR environment created", flush=True)

        def reset(self):
            print(
                f"[test] reset returns {len(self.observations)} BAR unit records",
                flush=True,
            )
            return self.observations, {"simulated": True}

        def step(self, action):
            assert isinstance(action, bar_ai.Action)
            self.received_actions.append(action)
            print(
                f"[test] BAR_Environment.step #{len(self.received_actions)}: "
                f"action_id={action.action_id}, unit_id={action.unit_id}",
                flush=True,
            )
            return (
                self.observations,
                0.1,
                False,
                False,
                {"own_alive_count": 3, "enemy_alive_count": 3},
            )

        def close(self):
            print("[test] simulated environment closed", flush=True)

    class SingleEpochTrainerArgs(train_fixed_v7.TrainerArgs):
        def __init__(self):
            super().__init__()
            self.ppo_epoch = 1

    class CountingRMAPPO(train_fixed_v7.R_MAPPO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.train_calls = 0
            trainers.append(self)

        def train(self, buffer, update_actor=True):
            self.train_calls += 1
            print("[test] starting the single MAPPO update", flush=True)
            return super().train(buffer, update_actor=update_actor)

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

            print("[test] launching train_fixed_v7 for one rollout step", flush=True)
            train_fixed_v7.main()

        assert len(env_instances) == 1
        assert len(env_instances[0].received_actions) == 3
        assert len(trainers) == 1
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

        print(
            "[test] passed: BAR observation dict -> train_fixed_v7 -> one MAPPO "
            "update -> three bar_ai.Action handoffs",
            flush=True,
        )


if __name__ == "__main__":
    test_train_fixed_v7_dataflow_one_mappo_update()