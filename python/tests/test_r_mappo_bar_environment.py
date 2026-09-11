from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.environment.bar_environment as bar_environment_module
from src.environment.bar_environment import BAR_Environment
from src.train.policy import R_MAPPO_Policy
from src.train.replay_buffer import SharedReplayBuffer
from src.train.r_mappo import R_MAPPO


class DummyUnit:
    """Minimal unit double exposing the BAR pawn inspection interface."""

    def __init__(self, unit_type, team, position, health):
        """Initializes unit type, team, position, and health fields."""
        self.unit_type = unit_type
        self.team = team
        self.position = position
        self.health = health

    def getUnitType(self):
        """Returns the mocked unit definition identifier."""
        return self.unit_type

    def getTeam(self):
        """Returns the mocked team identifier."""
        return self.team

    def getXPosition(self):
        """Returns the unit x-coordinate."""
        return self.position[0]

    def getYPosition(self):
        """Returns the unit y-coordinate."""
        return self.position[1]

    def getZPosition(self):
        """Returns the unit z-coordinate."""
        return self.position[2]

    def getHealth(self):
        """Returns the unit health value."""
        return self.health


class DummyPawn(DummyUnit):
    """Unit double with sight and pawn-specific metadata."""

    def __init__(self):
        """Initializes a pawn and nearby ally/enemy unit fixtures."""
        super().__init__(unit_type=1, team=0, position=(10.0, 20.0, 0.0), health=100.0)
        self.units_in_sight = [
            DummyUnit(2, 1, (12.0, 20.0, 0.0), 80.0),
            DummyUnit(2, 1, (8.0, 21.0, 0.0), 60.0),
            DummyUnit(2, 1, (10.0, 18.0, 0.0), 40.0),
            DummyUnit(1, 0, (11.0, 20.0, 0.0), 90.0),
            DummyUnit(1, 0, (9.0, 20.0, 0.0), 70.0),
        ]

    def getMaxHealth(self):
        """Returns the pawn maximum health."""
        return 100.0

    def getSightRange(self):
        """Returns the pawn sight radius."""
        return 500.0

    def getUnitsInSight(self):
        """Returns the configured nearby unit fixtures."""
        return self.units_in_sight


class DummyReader:
    """Reader double containing deterministic ally and enemy unit records."""

    def __init__(self):
        """Initializes deterministic own, enemy, and ally unit collections."""
        self.own_units = [
            SimpleNamespace(
                unit_id=index,
                unit_def_id=1,
                pos_x=10.0 + index,
                pos_y=20.0,
                pos_z=0.0,
                health=100.0,
                max_health=100.0,
                los_radius=500.0,
                is_dead=False,
                team_id=0,
            )
            for index in range(3)
        ]
        self.enemy_units = [
            SimpleNamespace(
                unit_id=10,
                unit_def_id=2,
                pos_x=12.0,
                pos_y=20.0,
                pos_z=0.0,
                health=80.0,
                team_id=1,
            )
        ]
        self.ally_units = [
            SimpleNamespace(
                unit_id=20,
                unit_def_id=1,
                pos_x=11.0,
                pos_y=20.0,
                pos_z=0.0,
                health=90.0,
                team_id=0,
            )
        ]

    def get_unit_by_id(self, unit_id):
        """Returns an own unit by index, or None when it is out of range."""
        return self.own_units[unit_id] if 0 <= unit_id < len(self.own_units) else None

    def get_enemy_units_in_sight(self, unit_id):
        """Returns the deterministic enemy sight fixtures."""
        return self.enemy_units

    def get_ally_units_in_sight(self, unit_id):
        """Returns the deterministic ally sight fixtures."""
        return self.ally_units


class DummySession:
    """Running session double exposing a deterministic unit reader."""

    def __init__(self):
        """Initializes the dummy reader."""
        self.reader = DummyReader()

    def is_running(self):
        """Reports that the dummy session is running."""
        return True

    def stop(self):
        """Stops the dummy session without external side effects."""
        pass


class DummyGRPCServer:
    """No-op gRPC server double for environment construction tests."""

    def start(self):
        """Starts the no-op server."""
        pass

    def stop(self, grace=2.0):
        """Stops the no-op server after the requested grace period."""
        pass


def trainer_args():
    """
    Builds the trainer configuration used by the integration test.

    Returns
    -------
    SimpleNamespace
        PPO settings accepted by `R_MAPPO`.
    """
    return SimpleNamespace(
        clip_param=0.2,
        ppo_epoch=2,
        num_mini_batch=1,
        data_chunk_length=1,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        huber_delta=10.0,
        use_recurrent_policy=False,
        use_naive_recurrent_policy=False,
        use_max_grad_norm=True,
        use_clipped_value_loss=True,
        use_huber_loss=False,
        use_popart=False,
        use_valuenorm=False,
        use_value_active_masks=True,
        use_policy_active_masks=True,
    )


def test_r_mappo_consumes_bar_environment_dummy_observations(monkeypatch):
    """
    Verifies that BAR observations can populate and train a MAPPO buffer.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace engine bindings with deterministic doubles.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If observation shapes, values, or training outputs are invalid.
    """
    print("=" * 60)
    print("TEST 7: BAR Environment -> Multi-Agent R_MAPPO Training")
    print("=" * 60)

    monkeypatch.setattr(
        bar_environment_module,
        "bar_ai",
        SimpleNamespace(UnitData=lambda: object(), Pawn=lambda data: DummyPawn()),
    )
    monkeypatch.setattr(
        bar_environment_module,
        "UnBARableAIGRPCServer",
        DummyGRPCServer,
    )

    env = BAR_Environment()
    env.session = DummySession()
    try:
        observations = np.asarray(env.get_obs(), dtype=np.float32)

        assert observations.shape == (3, 32)
        assert np.all(np.isfinite(observations))
        assert observations[0, 0] == 1.0
        assert observations[0, 7] == 2.0
        print("✓ BAR_Environment dummy observations received")
        print(f"  - All environment observations: {observations.shape}")
        print("  - Training agents: 2")
        print("  - Observation dimension: 32")

        device = torch.device("cpu")
        num_agents = 2
        action_dim = 5
        buffer_size = 4
        policy = R_MAPPO_Policy(obs_dim=32, action_dim=action_dim, device=device)
        trainer = R_MAPPO(trainer_args(), policy, device=device)
        buffer = SharedReplayBuffer(
            num_agents=num_agents,
            obs_shape=(32,),
            action_shape=(2,),
            action_dim=action_dim,
            buffer_size=buffer_size,
            device=device,
        )

        print("\nExpected buffer insert shapes:")
        print(f"  share_obs:        {buffer.share_obs[0].shape}")
        print(f"  obs:              {buffer.obs[0].shape}")
        print(f"  actions:          {buffer.actions[0].shape}")
        print(f"  action_log_probs: {buffer.action_log_probs[0].shape}")
        print(f"  value_preds:      {buffer.value_preds[0].shape}")
        print(f"  rewards:          {buffer.rewards[0].shape}")
        print(f"  masks:            {buffer.masks[0].shape}")
        print(f"  active_masks:     {buffer.active_masks[0].shape}")

        print("\nPopulating buffer with BAR_Environment observations...")
        for _ in range(buffer_size):
            rollout_observations = np.asarray(env.get_obs(), dtype=np.float32)
            agent_observations = rollout_observations[:num_agents]
            shared_observation = agent_observations.mean(axis=0)
            actions, log_probs = policy.get_action(
                torch.from_numpy(agent_observations)
            )
            values = policy.critic(torch.from_numpy(shared_observation)).detach()
            values = np.repeat(values.numpy().reshape(1, 1), num_agents, axis=0)

            buffer.insert(
                share_obs=shared_observation,
                obs=agent_observations,
                actions=actions.astype(np.float32),
                action_log_probs=log_probs.reshape(num_agents, 1).astype(np.float32),
                value_preds=values,
                rewards=np.ones((num_agents, 1), dtype=np.float32),
                masks=np.ones((num_agents, 1), dtype=np.float32),
                active_masks=np.ones((num_agents, 1), dtype=np.float32),
            )
        print("✓ Buffer populated")

        print("\nComputing returns...")
        next_observations = np.asarray(env.get_obs(), dtype=np.float32)[:num_agents]
        next_shared_observation = next_observations.mean(axis=0)
        next_value = policy.critic(
            torch.from_numpy(next_shared_observation)
        ).detach().numpy()
        buffer.compute_returns(
            np.repeat(next_value.reshape(1, 1), num_agents, axis=0).astype(np.float32)
        )
        print("✓ Returns computed")
        print("Buffer internal shapes after population:")
        print(f"  returns:      {buffer.returns.shape}")
        print(f"  value_preds:  {buffer.value_preds.shape}")

        print("\nRunning multi-agent training loop...")
        trainer.prep_training()
        train_info = trainer.train(buffer)

        assert set(train_info) == {
            "value_loss",
            "policy_loss",
            "dist_entropy",
            "actor_grad_norm",
            "critic_grad_norm",
            "ratio",
            "actions",
        }
        # Check numeric values are finite (skip "actions" list)
        numeric_keys = {"value_loss", "policy_loss", "dist_entropy", "actor_grad_norm", "critic_grad_norm", "ratio"}
        assert all(np.isfinite(train_info[key]) for key in numeric_keys)
        # Check actions is a list
        assert isinstance(train_info["actions"], list)
        assert buffer.obs.shape == (buffer_size + 1, num_agents, 32)
        print("✓ Multi-agent training completed")
        print(f"\nTraining Statistics (averaged over {trainer.ppo_epoch} epochs):")
        for key, value in train_info.items():
            if isinstance(value, list):
                print(f"  - {key:20s}: {len(value)} actions decoded")
            else:
                print(f"  - {key:20s}: {value:.6f}")
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))