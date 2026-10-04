"""Load a trained R_MAPPO checkpoint and run BAR episodes without training."""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch

from src.environment.bar_environment import BAR_Environment
from src.train.policy import R_MAPPO_Policy
from src.train.obs_r_mappo import OBS_DIM
from src.train.train_fixed_v4 import (
    _make_bar_action,
    _parse_reset_result,
    _parse_step_result,
    _policy_sample_actions,
    _prepare_available_actions,
    _prepare_obs,
    _prepare_share_obs,
)


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[R_MAPPO_Policy, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy = R_MAPPO_Policy(
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_dim"]),
        device=device,
        lr=0.0,
    )
    policy.actor.load_state_dict(checkpoint["actor_state_dict"])
    policy.critic.load_state_dict(checkpoint["critic_state_dict"])
    policy.actor.eval()
    policy.critic.eval()
    return policy, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained R_MAPPO BAR agent")
    parser.add_argument("--model", type=Path, default=Path("models/r_mappo/r_mappo_latest.pt"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--training-team-id", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, checkpoint = load_policy(args.model, device)
    num_agents = int(checkpoint.get("num_agents", 3))
    obs_dim = int(checkpoint.get("obs_dim", OBS_DIM))
    action_dim = int(checkpoint.get("action_dim", 5))
    training_team_id = (
        int(args.training_team_id)
        if args.training_team_id is not None
        else int(checkpoint.get("training_team_id", 0))
    )

    print(f"Loaded {args.model} (trained through episode {checkpoint.get('episode', '?')})")

    for episode in range(args.episodes):
        # BAR_Environment currently cannot restart a stopped gRPC server, so
        # inference also uses one fresh environment per episode.
        env = BAR_Environment()
        episode_reward = 0.0
        try:
            obs_raw, share_raw, available_raw, _ = _parse_reset_result(env.reset())
            obs = _prepare_obs(obs_raw, num_agents, obs_dim, training_team_id)
            share_obs = _prepare_share_obs(share_raw, obs, obs_dim)
            available = _prepare_available_actions(available_raw, num_agents, action_dim)

            for step in range(args.max_steps):
                with torch.no_grad():
                    _, actions, _ = _policy_sample_actions(
                        policy, obs, share_obs, available, device, num_agents
                    )

                # BAR_Environment.step accepts one Action and advances one
                # engine update, so issue each controlled unit's command.
                final_result = None
                step_reward = 0.0
                for action_index, unit_id in np.asarray(actions).reshape(num_agents, 2):
                    final_result = env.step(_make_bar_action(action_index, unit_id))
                    step_reward += float(np.asarray(final_result[1]).sum())
                    if bool(final_result[2]) or bool(final_result[3]):
                        break

                if final_result is None:
                    raise RuntimeError("Policy produced no actions")

                next_obs_raw, next_share_raw, reward, terminated, truncated, info, next_available_raw = _parse_step_result(final_result)
                episode_reward += step_reward
                print(f"episode={episode + 1} step={step + 1} actions={actions.tolist()} reward={step_reward:.4f}")

                if bool(np.all(np.asarray(terminated))) or bool(np.all(np.asarray(truncated))):
                    print(f"Episode ended: terminated={terminated}, truncated={truncated}, info={info}")
                    break

                obs = _prepare_obs(next_obs_raw, num_agents, obs_dim, training_team_id)
                share_obs = _prepare_share_obs(next_share_raw, obs, obs_dim)
                available = _prepare_available_actions(next_available_raw, num_agents, action_dim)

            print(f"Episode {episode + 1}: total reward={episode_reward:.4f}")
        finally:
            try:
                env.close()
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    main()
