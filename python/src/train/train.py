"""
Training script integrating BAR environment, policy, replay buffer, and R_MAPPO trainer.

Empfohlener Start (wenn diese Datei als `python/train.py` gespeichert ist):
    uv run --active python train.py

Oder mit Parametern:
    uv run --active python train.py --num-episodes 100 --num-mini-batch 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from src.environment.bar_environment import BAR_Environment, EngineSessionConfig
from src.train.policy import R_MAPPO_Policy
from src.train.replay_buffer import SharedReplayBuffer
from src.train.r_mappo import R_MAPPO


# -----------------------------------------------------------------------------
# Trainer-Args
# -----------------------------------------------------------------------------
class TrainerArgs:
    """Config namespace for R_MAPPO trainer."""

    def __init__(self) -> None:
        self.clip_param = 0.2
        self.ppo_epoch = 10
        self.num_mini_batch = 4
        self.data_chunk_length = 4
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.huber_delta = 10.0

        self.use_recurrent_policy = False
        self.use_naive_recurrent_policy = False
        self.use_max_grad_norm = True
        self.use_clipped_value_loss = True
        self.use_huber_loss = False
        self.use_popart = False
        self.use_valuenorm = False
        self.use_value_active_masks = True
        self.use_policy_active_masks = True


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def _as_bool_done(x: Any) -> bool:
    """
    Wandelt terminated/truncated robust in ein einzelnes bool um.
    Funktioniert für bool, Listen, Tupel, numpy arrays.
    """
    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    arr = np.asarray(x)
    if arr.size == 0:
        return False
    return bool(arr.all())


def _prepare_obs(obs: Any, num_agents: int, obs_dim: int) -> np.ndarray:
    """
    Bringt Beobachtungen robust in Shape: (num_agents, obs_dim)
    """
    if obs is None:
        return np.zeros((num_agents, obs_dim), dtype=np.float32)

    arr = np.asarray(obs, dtype=np.float32)

    # Falls 1D -> versuchen, auf (num_agents, obs_dim) zu bringen
    if arr.ndim == 1:
        if arr.size == obs_dim:
            arr = np.tile(arr[None, :], (num_agents, 1))
        elif arr.size == num_agents * obs_dim:
            arr = arr.reshape(num_agents, obs_dim)
        else:
            flat = np.zeros((num_agents * obs_dim,), dtype=np.float32)
            n = min(flat.size, arr.size)
            flat[:n] = arr.reshape(-1)[:n]
            arr = flat.reshape(num_agents, obs_dim)

    # Falls mehrdimensional, erste Achse als Agenten interpretieren
    elif arr.ndim >= 2:
        if arr.shape[0] != num_agents:
            flat = arr.reshape(-1)
            padded = np.zeros((num_agents * obs_dim,), dtype=np.float32)
            n = min(padded.size, flat.size)
            padded[:n] = flat[:n]
            arr = padded.reshape(num_agents, obs_dim)
        else:
            arr = arr.reshape(num_agents, -1)
            current_dim = arr.shape[1]
            if current_dim != obs_dim:
                fixed = np.zeros((num_agents, obs_dim), dtype=np.float32)
                n = min(obs_dim, current_dim)
                fixed[:, :n] = arr[:, :n]
                arr = fixed

    return arr.astype(np.float32)


def _prepare_reward(reward: Any, num_agents: int) -> np.ndarray:
    """
    Bringt Reward robust in Shape: (num_agents, 1)
    """
    if reward is None:
        return np.zeros((num_agents, 1), dtype=np.float32)

    arr = np.asarray(reward, dtype=np.float32)

    if arr.ndim == 0:
        arr = np.full((num_agents, 1), float(arr), dtype=np.float32)
    elif arr.ndim == 1:
        if arr.size == 1:
            arr = np.full((num_agents, 1), float(arr[0]), dtype=np.float32)
        elif arr.size == num_agents:
            arr = arr.reshape(num_agents, 1)
        else:
            fixed = np.zeros((num_agents,), dtype=np.float32)
            n = min(num_agents, arr.size)
            fixed[:n] = arr[:n]
            arr = fixed.reshape(num_agents, 1)
    else:
        arr = arr.reshape(num_agents, -1)
        if arr.shape[1] != 1:
            arr = arr[:, :1]

    return arr.astype(np.float32)


def _repeat_shared_obs(shared_obs: np.ndarray, num_agents: int) -> np.ndarray:
    """
    Macht aus (obs_dim,) -> (num_agents, obs_dim)
    """
    return np.repeat(shared_obs[None, :], num_agents, axis=0).astype(np.float32)


def _repeat_value(value_tensor: torch.Tensor, num_agents: int) -> np.ndarray:
    """
    Macht aus Critic-Output einen Buffer-kompatiblen Array.
    Zielshape: (num_agents, 1)
    """
    value_np = value_tensor.detach().cpu().numpy().reshape(-1)

    # Falls der Critic nur einen Scalar zurückgibt -> an alle Agenten replizieren
    if value_np.size == 1:
        return np.full((num_agents, 1), float(value_np[0]), dtype=np.float32)

    # Falls mehrere Werte kommen -> passend abschneiden/auffüllen
    fixed = np.zeros((num_agents,), dtype=np.float32)
    n = min(num_agents, value_np.size)
    fixed[:n] = value_np[:n]
    return fixed.reshape(num_agents, 1)


# -----------------------------------------------------------------------------
# Hauptprogramm
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train BAR environment with R_MAPPO")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--num-mini-batch", type=int, default=4, help="Number of mini-batches")
    parser.add_argument("--buffer-size", type=int, default=256, help="Replay buffer size")
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=128,
        help="Observation dimension (Fallback, falls nicht aus Env ableitbar)",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Action dimension (Fallback, falls nicht aus Env ableitbar)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print("Initializing BAR environment...")
    env_config = EngineSessionConfig()
    env = BAR_Environment(env_config)

    # -------------------------------------------------------------------------
    # Dimensions-Fallbacks
    # Hinweis: Wenn deine Env observation/action spaces anbietet, solltest du
    # das hier langfristig sauber aus der Env ableiten.
    # -------------------------------------------------------------------------
    obs_dim = args.obs_dim
    action_dim = args.action_dim

    print(f"Using obs_dim={obs_dim}, action_dim={action_dim}, num_agents={args.num_agents}")
    print("Initializing policy...")
    policy = R_MAPPO_Policy(obs_dim, action_dim, device=device, lr=args.lr)

    print("Initializing replay buffer...")
    buffer = SharedReplayBuffer(
        num_agents=args.num_agents,
        obs_shape=(obs_dim,),
        action_shape=(1,),
        buffer_size=args.buffer_size,
        device=device,
    )

    print("Initializing R_MAPPO trainer...")
    trainer_args = TrainerArgs()
    trainer_args.num_mini_batch = args.num_mini_batch
    trainer = R_MAPPO(trainer_args, policy, device=device)

    print("\nStarting training loop...")

    try:
        for episode in range(args.num_episodes):
            # -------------------------------------------------------------
            # Reset
            # -------------------------------------------------------------
            reset_result = env.reset()

            if isinstance(reset_result, tuple) and len(reset_result) >= 1:
                obs = reset_result[0]
                info = reset_result[1] if len(reset_result) > 1 else {}
            else:
                obs = reset_result
                info = {}

            obs = _prepare_obs(obs, args.num_agents, obs_dim)
            share_obs = obs.mean(axis=0).astype(np.float32)

            episode_reward = 0.0
            done = False
            step_count = 0

            # -------------------------------------------------------------
            # Rollout
            # -------------------------------------------------------------
            trainer.prep_rollout()

            while not done and step_count < args.buffer_size:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

                with torch.no_grad():
                    # Actor -> diskrete Aktion
                    action_logits = policy.actor(obs_tensor)
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    actions = action_dist.sample()
                    action_log_probs = action_dist.log_prob(actions)

                    # Zentralisierter Critic
                    share_obs_tensor = torch.as_tensor(
                        share_obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    values = policy.critic(share_obs_tensor)

                # ---------------------------------------------------------
                # Environment step
                # ---------------------------------------------------------
                step_result = env.step(actions.detach().cpu().numpy())

                if not isinstance(step_result, tuple) or len(step_result) < 5:
                    raise RuntimeError(
                        "env.step(...) muss ein Tupel liefern wie "
                        "(obs, reward, terminated, truncated, info)"
                    )

                next_obs, reward, terminated, truncated, step_info = step_result

                done = _as_bool_done(terminated) or _as_bool_done(truncated)

                next_obs = _prepare_obs(next_obs, args.num_agents, obs_dim)
                next_share_obs = next_obs.mean(axis=0).astype(np.float32)
                reward_arr = _prepare_reward(reward, args.num_agents)

                value_preds = _repeat_value(values, args.num_agents)
                share_obs_repeated = _repeat_shared_obs(share_obs, args.num_agents)

                masks = np.ones((args.num_agents, 1), dtype=np.float32) * (0.0 if done else 1.0)
                active_masks = np.ones((args.num_agents, 1), dtype=np.float32)

                # ---------------------------------------------------------
                # Buffer insert
                # ---------------------------------------------------------
                buffer.insert(
                    share_obs=share_obs_repeated,
                    obs=obs,
                    actions=actions.detach().cpu().numpy().reshape(args.num_agents, 1),
                    action_log_probs=action_log_probs.detach().cpu().numpy().reshape(args.num_agents, 1),
                    value_preds=value_preds,
                    rewards=reward_arr,
                    masks=masks,
                    active_masks=active_masks,
                )

                episode_reward += float(reward_arr.mean())
                step_count += 1

                # State weiterschieben
                obs = next_obs
                share_obs = next_share_obs

            # -------------------------------------------------------------
            # Bootstrap für Returns
            # -------------------------------------------------------------
            with torch.no_grad():
                share_obs_tensor = torch.as_tensor(
                    share_obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                next_value_tensor = policy.critic(share_obs_tensor)

            next_value = _repeat_value(next_value_tensor, args.num_agents)

            # Manche Implementierungen erwarten exakt dieses Argument
            buffer.compute_returns(next_value, gamma=args.gamma)

            # -------------------------------------------------------------
            # Training
            # -------------------------------------------------------------
            trainer.prep_training()
            train_info = trainer.train(buffer, update_actor=True)

            # Buffer zurücksetzen
            buffer.reset()

            # Logging
            print(f"Episode {episode + 1}/{args.num_episodes}")
            print(f"  Episode Reward: {episode_reward:.4f}")
            print(f"  Steps: {step_count}")
            print(f"  Value Loss: {train_info.get('value_loss', 0):.6f}")
            print(f"  Policy Loss: {train_info.get('policy_loss', 0):.6f}")
            print(f"  Entropy: {train_info.get('dist_entropy', 0):.6f}")
            print()

    finally:
        env.close()
        print("Training complete!")


if __name__ == "__main__":
    main()
