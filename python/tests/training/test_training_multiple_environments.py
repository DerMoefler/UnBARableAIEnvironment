import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from src.environment import engine_pool as engine_pool_module
from src.environment.bar_environment import BAR_Environment
from src.environment.stepinfo import StepInfo
from src.train import training


def test_training_uses_two_environments_and_two_dummy_engines(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = {
        "run_name": "two_environment_test",
        "model_name": "model.pt",
        "algorithm": "MAPPO",
        "device": "cpu",
        "number_of_agents": 1,
        "number_of_enemies": 1,
        "action_dimension": 2,
        "self_play": False,
        "adapter": {},
        "training": {
            "number_of_environments": 2,
            "number_of_engines": 2,
            "maximum_steps": 1,
            "buffer_size": 1,
            "continue_training": False,
            "mini_batch": 1,
            "episodes": 1,
            "save_every_x_episode": 0,
            "mappo": {"gamma": 0.99},
        },
    }
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    created_engine_ids: list[int] = []
    environments: list[Any] = []
    adapters: list[Any] = []
    trainer_instances: list[Any] = []
    runner = DummyRunner()

    class DummyEngineSession:
        def __init__(self, engine_id: int) -> None:
            self.engine_id = engine_id
            self.shared_memories = {
                0: SimpleNamespace(
                    get_engine_status=lambda: engine_pool_module.bar_ai.EngineStatus.READY
                )
            }

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def is_running(self) -> bool:
            return True

        def wait_for_engine_status(self, status, timeout: float) -> bool:
            return True

        def get_shared_memory(self, team_id: int):
            return self.shared_memories[team_id]

        def get_engine_id(self) -> int:
            return self.engine_id

    class DummyEngineSessionFactory:
        def __init__(self, **kwargs) -> None:
            pass

        def __call__(self, engine_id: int) -> DummyEngineSession:
            created_engine_ids.append(engine_id)
            return DummyEngineSession(engine_id)

    class DummyEnvironment:
        def __init__(self, engine_pool, environment_id: int) -> None:
            self.engine_pool = engine_pool
            self.session = engine_pool.acquire(timeout=2.0)
            self.environment_id = environment_id
            self.unbarable_ai_team_ids = (0,)
            self.current_team_id = 0
            self.episode_step = 0
            self.stepinfo = StepInfo(action_dim=2)
            self.step_calls = 0
            environments.append(self)

        def reset(self):
            self.current_team_id = 0
            self.episode_step = 0
            self.step_calls = 0
            self.stepinfo = StepInfo(action_dim=2)
            return {}, {"current_team_id": 0}

        def step(self, actions: list[Any]):
            self.step_calls += 1
            self.episode_step += 1
            return (
                {},
                1.0,
                True,
                False,
                {
                    "acting_team_id": 0,
                    "current_team_id": 0,
                    "reward_team_id": 0,
                },
            )

        def close(self) -> None:
            pass

    class DummyAdapter:
        num_agents = 1
        obs_dim = 2
        action_dim = 2
        target_feature_dim = 1
        training_team_id = 0

        def __init__(self, environment_id: int) -> None:
            self.environment_id = environment_id
            self.agent_unit_ids = [environment_id + 1]

        def assign_adapter_id(self, adapter_id: int) -> None:
            self.adapter_id = adapter_id

        def reset_episode(self, obs_raw) -> None:
            pass

        def prepare(self, obs_raw):
            return SimpleNamespace(
                environment_id=self.environment_id,
                share_obs=np.zeros(2, dtype=np.float32),
                obs=np.zeros((1, 2), dtype=np.float32),
                active_mask=np.ones((1, 1), dtype=np.float32),
                available_actions=np.ones((1, 2), dtype=np.float32),
                target_features=None,
                target_mask=None,
            )

        def prepare_reward(self, reward):
            return np.full((1, 1), reward, dtype=np.float32)

    class DummyTrainer:
        def __init__(self, *args, **kwargs) -> None:
            self.buffers = []
            trainer_instances.append(self)

        def prep_rollout(self) -> None:
            pass

        def prep_training(self) -> None:
            pass

        def train_multiple(self, buffers, update_actor=True):
            self.buffers = list(buffers)
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "dist_entropy": 0.0,
            }

    def create_adapters(number_of_environments, data, *, team_id):
        created = [
            DummyAdapter(environment_id)
            for environment_id in range(number_of_environments)
        ]
        adapters.extend(created)
        return created

    def create_environment(config_path, engine_pool, *, environment_id):
        return DummyEnvironment(engine_pool, environment_id)

    monkeypatch.setattr(
        engine_pool_module,
        "EngineSessionFactory",
        DummyEngineSessionFactory,
    )
    monkeypatch.setattr(
        BAR_Environment,
        "from_config",
        classmethod(
            lambda cls, config_path, engine_pool, *, environment_id:
                create_environment(
                    config_path,
                    engine_pool,
                    environment_id=environment_id,
                )
        ),
    )
    monkeypatch.setattr(
        training,
        "configure_logging",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        training,
        "_select_device",
        lambda config: torch.device("cpu"),
    )
    monkeypatch.setattr(
        training,
        "load_training_config",
        lambda path: (
            config,
            config["training"],
            config["training"]["mappo"],
        ),
    )
    monkeypatch.setattr(
        training,
        "create_mappo_adapters",
        create_adapters,
    )
    monkeypatch.setattr(
        training,
        "MAPPOPolicyRunner",
        SimpleNamespace(
            from_config=lambda *args, **kwargs: runner
        ),
    )
    monkeypatch.setattr(training, "R_MAPPO", DummyTrainer)
    monkeypatch.setattr(
        training,
        "_save_checkpoint",
        lambda policy, adapter, episode, model_dir, *, model_name:
            model_dir / model_name,
    )

    training.main(config_path=config_path)

    assert len(environments) == 2
    assert {environment.environment_id for environment in environments} == {0, 1}
    assert {environment.session.get_engine_id() for environment in environments} == {0, 1}
    assert environments[0].engine_pool.pool_size == 2
    assert set(created_engine_ids) == {0, 1}
    assert [environment.step_calls for environment in environments] == [1, 1]
    assert [adapter.environment_id for adapter in adapters] == [0, 1]
    assert runner.acted_environment_ids == [0, 1]
    assert len(trainer_instances) == 1
    assert len(trainer_instances[0].buffers) == 1
    assert trainer_instances[0].buffers[0].step == 2


class DummyRunner:
    def __init__(self) -> None:
        self.policy = object()
        self.acted_environment_ids: list[int] = []

    def set_evaluation_mode(self) -> None:
        pass

    def set_training_mode(self) -> None:
        pass

    def act(self, prepared):
        self.acted_environment_ids.append(prepared.environment_id)
        return SimpleNamespace(
            actions=np.zeros((1, 2), dtype=np.float32),
            action_log_probs=np.zeros((1, 1), dtype=np.float32),
            value_predictions=np.zeros((1, 1), dtype=np.float32),
        )

    def create_bar_actions(self, policy_step, prepared, *, team_id: int):
        return []
