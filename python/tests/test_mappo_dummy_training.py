from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from src.train.mappo.obs_r_mappo import OBS_DIM, build_r_mappo_observations
from src.train.mappo.r_mappo import R_MAPPO
from src.train.policy import Actor, Critic, R_MAPPO_Policy
from src.train.replay_buffer import SharedReplayBuffer


NUM_AGENTS = 3
ACTION_DIM = 5
TARGET_DIM = 3
ROLLOUT_STEPS = 10
ACTION_NAMES = ("north", "south", "east", "west", "attack")


class _TestSGD:
    def __init__(self, parameters, learning_rate: float):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-self.learning_rate)


class TrainerArgs:
    clip_param = 0.2
    ppo_epoch = 1
    num_mini_batch = 1
    data_chunk_length = 4
    value_loss_coef = 1.0
    entropy_coef = 0.05
    max_grad_norm = 0.5
    huber_delta = 10.0
    use_recurrent_policy = False
    use_naive_recurrent_policy = False
    use_max_grad_norm = True
    use_clipped_value_loss = True
    use_huber_loss = False
    use_popart = False
    use_valuenorm = False
    use_value_active_masks = True
    use_policy_active_masks = True


def _dummy_units(step: int) -> dict[str, SimpleNamespace]:
    units = {}
    formations = (
        (0, 0, 130.0, 210.0, 0.0, 1.00),
        (1, 0, 170.0, 245.0, 0.0, 0.82),
        (2, 0, 205.0, 190.0, 0.0, 0.64),
        (3, 1, 610.0, 540.0, 0.0, 0.92),
        (4, 1, 665.0, 505.0, 0.0, 0.74),
        (5, 1, 700.0, 570.0, 0.0, 0.56),
    )
    for unit_id, team_id, x, z, y, health_fraction in formations:
        drift = step * (2.0 if team_id == 0 else -1.5)
        health = max(1.0, 100.0 * health_fraction - step * (unit_id % 3))
        units[str(unit_id)] = SimpleNamespace(
            unit_id=unit_id,
            team_id=team_id,
            ally_team_id=team_id,
            unit_def_id=102,
            health=health,
            max_health=100.0,
            pos_x=x + drift,
            pos_y=y,
            pos_z=z + drift,
            los_radius=350.0,
            is_dead=False,
            being_built=False,
        )
    return units


def test_mappo_trains_on_dummy_observations():
    print("Synthetic MAPPO test (CPU, no environment)", flush=True)
    torch.manual_seed(7)
    np.random.seed(7)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    print("Initializing policy networks...", flush=True)
    policy = R_MAPPO_Policy.__new__(R_MAPPO_Policy)
    policy.device = device
    policy.obs_dim = OBS_DIM
    policy.action_dim = ACTION_DIM
    policy.target_dim = TARGET_DIM
    policy.actor = Actor(OBS_DIM, ACTION_DIM).to(device)
    policy.target_actor = Actor(OBS_DIM, TARGET_DIM).to(device)
    policy.critic = Critic(OBS_DIM).to(device)
    policy.actor_optimizer = _TestSGD(
        list(policy.actor.parameters()) + list(policy.target_actor.parameters()),
        learning_rate=1e-4,
    )
    policy.critic_optimizer = _TestSGD(
        policy.critic.parameters(), learning_rate=1e-4
    )
    with torch.no_grad():
        for actor in (policy.actor, policy.target_actor):
            torch.nn.init.zeros_(actor.net[-1].weight)
            torch.nn.init.zeros_(actor.net[-1].bias)

    print("Policy ready; creating trainer and replay buffer...", flush=True)
    trainer = R_MAPPO(TrainerArgs(), policy, device=device)
    buffer = SharedReplayBuffer(
        num_agents=NUM_AGENTS,
        obs_shape=(OBS_DIM,),
        action_shape=(2,),
        buffer_size=ROLLOUT_STEPS,
        device=device,
        action_dim=ACTION_DIM,
    )
    recent_actions = deque(maxlen=10)
    sampled_action_types = set()
    sampled_target_ids = set()
    print(
        f"Setup passed: {NUM_AGENTS} agents, {OBS_DIM} observation features, "
        f"{ROLLOUT_STEPS} rollout steps",
        flush=True,
    )

    trainer.prep_rollout()
    for step in range(ROLLOUT_STEPS):
        observations = build_r_mappo_observations(_dummy_units(step))
        assert observations.shape == (NUM_AGENTS, OBS_DIM)
        assert np.isfinite(observations).all()
        if step == 0:
            print("Synthetic input summary:", flush=True)
            for agent_id, observation in enumerate(observations):
                ally_count = int(observation[8]) + int(observation[17])
                enemy_count = sum(
                    int(observation[26 + slot * 9]) for slot in range(3)
                )
                print(
                    f"  Agent {agent_id}: unit type={int(observation[0])}, "
                    f"health={observation[1]:.2f}, "
                    f"position=({observation[2]:.3f}, {observation[3]:.3f}, "
                    f"{observation[4]:.3f}), allies={ally_count}, "
                    f"enemies={enemy_count}",
                    flush=True,
                )

        shared_observation = observations.mean(axis=0).astype(np.float32)
        observation_tensor = torch.as_tensor(observations, device=device)
        actions, action_log_probs = policy.get_action(observation_tensor)
        actions = np.asarray(actions, dtype=np.float32)
        action_log_probs = np.asarray(action_log_probs, dtype=np.float32).reshape(
            NUM_AGENTS, 1
        )
        assert actions.shape == (NUM_AGENTS, 2)
        assert np.all((0 <= actions[:, 0]) & (actions[:, 0] < ACTION_DIM))
        assert np.all((0 <= actions[:, 1]) & (actions[:, 1] < TARGET_DIM))
        sampled_action_types.update(actions[:, 0].astype(int).tolist())
        sampled_target_ids.update(actions[:, 1].astype(int).tolist())

        with torch.no_grad():
            values = policy.critic(
                torch.as_tensor(shared_observation, device=device).unsqueeze(0)
            ).cpu().numpy()
        value_predictions = np.repeat(values, NUM_AGENTS, axis=0).astype(np.float32)

        enemy_health = max(0.1, 0.75 - step * 0.025)
        rewards = np.asarray(
            [
                [0.1 * enemy_health if int(action[0]) == 4 else -0.01]
                for action in actions
            ],
            dtype=np.float32,
        )
        masks = np.ones((NUM_AGENTS, 1), dtype=np.float32)
        available_actions = np.ones((NUM_AGENTS, ACTION_DIM), dtype=np.float32)

        buffer.insert(
            shared_observation,
            observations,
            actions,
            action_log_probs,
            value_predictions,
            rewards,
            masks,
            masks,
            available_actions,
        )
        for agent_id, (action_type, target_id) in enumerate(actions):
            recent_actions.append(
                (
                    step + 1,
                    agent_id,
                    ACTION_NAMES[int(action_type)],
                    int(target_id),
                )
            )
        print(f"Step {step + 1} passed", flush=True)

    with torch.no_grad():
        next_value = policy.critic(
            torch.as_tensor(shared_observation, device=device).unsqueeze(0)
        ).cpu().numpy()
    buffer.compute_returns(np.repeat(next_value, NUM_AGENTS, axis=0).astype(np.float32))

    trainer.prep_training()
    train_info = trainer.train(buffer)

    print("Last 10 actions:", flush=True)
    for step, agent_id, action_name, target_id in recent_actions:
        print(
            f"  Step {step}, agent {agent_id}: {action_name}, target {target_id}",
            flush=True,
        )
    training_output = {
        name: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        for name, value in train_info.items()
    }
    assert buffer.step == 0
    assert np.isfinite(buffer.returns).all()
    assert all(np.isfinite(value) for value in training_output.values())
    assert len(recent_actions) == 10
    assert len(sampled_action_types) > 1
    assert len(sampled_target_ids) > 1
    assert train_info["dist_entropy"] > 0.0
    print("PPO update passed", flush=True)
    print(f"  Policy loss:       {training_output['policy_loss']:.5f}", flush=True)
    print(f"  Value loss:        {training_output['value_loss']:.5f}", flush=True)
    print(f"  Entropy:           {training_output['dist_entropy']:.5f}", flush=True)
    print(f"  Importance ratio:  {training_output['ratio']:.5f}", flush=True)
    print("All synthetic MAPPO checks passed.", flush=True)


if __name__ == "__main__":
    test_mappo_trains_on_dummy_observations()