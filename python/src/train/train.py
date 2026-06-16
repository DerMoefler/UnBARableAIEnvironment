"""
Training script integrating BAR environment, policy, replay buffer, and R_MAPPO trainer.

Empfohlener Start:
    PYTHONUNBUFFERED=1 uv run src/train/train.py

Oder alternativ:
    uv run python -u src/train/train.py

Oder mit Parametern:
    PYTHONUNBUFFERED=1 uv run src/train/train.py --num-episodes 100 --num-mini-batch 4
"""

from __future__ import annotations

print("DEBUG: script started", flush=True)

import argparse
import traceback
from typing import Any

print("DEBUG: stdlib imports done", flush=True)

import numpy as np
print("DEBUG: numpy imported", flush=True)

import torch
print("DEBUG: torch imported", flush=True)

from src.environment.bar_environment import BAR_Environment, EngineSessionConfig
print("DEBUG: bar_environment imported", flush=True)

from src.train.policy import R_MAPPO_Policy
print("DEBUG: policy imported", flush=True)

from src.train.replay_buffer import SharedReplayBuffer
print("DEBUG: replay_buffer imported", flush=True)

from src.train.r_mappo import R_MAPPO
print("DEBUG: r_mappo imported", flush=True)


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
    print("DEBUG: entered main()", flush=True)

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
    parser.add_argument(
        "--debug-shapes",
        action="store_true",
        help="Print tensor/array shapes during rollout",
    )

    args = parser.parse_args()
    print("DEBUG: args parsed", flush=True)
    print(f"DEBUG: args = {args}", flush=True)

    device = torch.device(args.device)
    print(f"DEBUG: device = {device}", flush=True)

    success = False
    env = None

    try:
        print("Initializing BAR environment...", flush=True)
        env_config = EngineSessionConfig()
        print("DEBUG: EngineSessionConfig created", flush=True)

        env = BAR_Environment(env_config)
        print("DEBUG: BAR_Environment created", flush=True)

        # ---------------------------------------------------------------------
        # Dimensions-Fallbacks
        # ---------------------------------------------------------------------
        obs_dim = args.obs_dim
        action_dim = args.action_dim

        print(
            f"Using obs_dim={obs_dim}, action_dim={action_dim}, num_agents={args.num_agents}",
            flush=True,
        )

        print("Initializing policy...", flush=True)
        policy = R_MAPPO_Policy(obs_dim, action_dim, device=device, lr=args.lr)
        print("DEBUG: policy created", flush=True)

        print("Initializing replay buffer...", flush=True)
        buffer = SharedReplayBuffer(
            num_agents=args.num_agents,
            obs_shape=(obs_dim,),
            action_shape=(1,),
            buffer_size=args.buffer_size,
            device=device,
        )
        print("DEBUG: replay buffer created", flush=True)

        print("Initializing R_MAPPO trainer...", flush=True)
        trainer_args = TrainerArgs()
        trainer_args.num_mini_batch = args.num_mini_batch
        trainer = R_MAPPO(trainer_args, policy, device=device)
        print("DEBUG: trainer created", flush=True)

        print("\nStarting training loop...", flush=True)

        for episode in range(args.num_episodes):
            print(f"DEBUG: starting episode {episode + 1}", flush=True)

            # -------------------------------------------------------------
            # Reset
            # -------------------------------------------------------------
            reset_result = env.reset()
            print("DEBUG: env.reset() returned", flush=True)

            if isinstance(reset_result, tuple) and len(reset_result) >= 1:
                obs = reset_result[0]
                info = reset_result[1] if len(reset_result) > 1 else {}
            else:
                obs = reset_result
                info = {}

            obs = _prepare_obs(obs, args.num_agents, obs_dim)

            # zentrale Observation für Critic und Buffer: Shape (obs_dim,)
            share_obs = obs.mean(axis=0).astype(np.float32)

            if args.debug_shapes:
                print(f"DEBUG: reset obs shape = {obs.shape}", flush=True)
                print(f"DEBUG: reset share_obs shape = {share_obs.shape}", flush=True)

            episode_reward = 0.0
            done = False
            step_count = 0

            # -------------------------------------------------------------
            # Rollout
            # -------------------------------------------------------------
            trainer.prep_rollout()
            print("DEBUG: trainer.prep_rollout() done", flush=True)

            while not done and step_count < args.buffer_size:
                if args.debug_shapes:
                    print(f"DEBUG: rollout step {step_count}", flush=True)

                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

                with torch.no_grad():
                    # Actor -> diskrete Aktion pro Agent
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
                env_actions = actions.detach().cpu().numpy()
                if args.debug_shapes:
                    print(f"DEBUG: env_actions shape = {env_actions.shape}", flush=True)

                step_result = env.step(env_actions)

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

                masks = np.ones((args.num_agents, 1), dtype=np.float32) * (
                    0.0 if done else 1.0
                )
                active_masks = np.ones((args.num_agents, 1), dtype=np.float32)

                if args.debug_shapes:
                    print(f"DEBUG: obs shape = {obs.shape}", flush=True)
                    print(f"DEBUG: share_obs shape = {share_obs.shape}", flush=True)
                    print(
                        f"DEBUG: buffer.share_obs[buffer.step] shape = "
                        f"{buffer.share_obs[buffer.step].shape}",
                        flush=True,
                    )
                    print(f"DEBUG: actions shape = {env_actions.reshape(args.num_agents, 1).shape}", flush=True)
                    print(
                        f"DEBUG: action_log_probs shape = "
                        f"{action_log_probs.detach().cpu().numpy().reshape(args.num_agents, 1).shape}",
                        flush=True,
                    )
                    print(f"DEBUG: value_preds shape = {value_preds.shape}", flush=True)
                    print(f"DEBUG: reward_arr shape = {reward_arr.shape}", flush=True)
                    print(f"DEBUG: masks shape = {masks.shape}", flush=True)
                    print(f"DEBUG: active_masks shape = {active_masks.shape}", flush=True)

                # ---------------------------------------------------------
                # Buffer insert
                # WICHTIG:
                # share_obs bleibt (obs_dim,) und wird NICHT wiederholt
                # ---------------------------------------------------------
                buffer.insert(
                    share_obs=share_obs,
                    obs=obs,
                    actions=env_actions.reshape(args.num_agents, 1),
                    action_log_probs=action_log_probs.detach().cpu().numpy().reshape(
                        args.num_agents, 1
                    ),
                    value_preds=value_preds,
                    rewards=reward_arr,
                    masks=masks,
                    active_masks=active_masks,
                )

                episode_reward += float(reward_arr.mean())
                step_count += 1

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

            if args.debug_shapes:
                print(f"DEBUG: next_value shape = {next_value.shape}", flush=True)

            buffer.compute_returns(next_value, gamma=args.gamma)
            print("DEBUG: buffer.compute_returns() done", flush=True)

            # -------------------------------------------------------------
            # Training
            # -------------------------------------------------------------
            trainer.prep_training()
            print("DEBUG: trainer.prep_training() done", flush=True)

            train_info = trainer.train(buffer, update_actor=True)
            print("DEBUG: trainer.train() done", flush=True)

            buffer.reset()
            print("DEBUG: buffer.reset() done", flush=True)

            print(f"Episode {episode + 1}/{args.num_episodes}", flush=True)
            print(f"  Episode Reward: {episode_reward:.4f}", flush=True)
            print(f"  Steps: {step_count}", flush=True)
            print(f"  Value Loss: {train_info.get('value_loss', 0):.6f}", flush=True)
            print(f"  Policy Loss: {train_info.get('policy_loss', 0):.6f}", flush=True)
            print(f"  Entropy: {train_info.get('dist_entropy', 0):.6f}", flush=True)
            print("", flush=True)

        success = True

    except Exception as e:
        print("\nERROR: Exception occurred during training!", flush=True)
        print(f"ERROR TYPE: {type(e).__name__}", flush=True)
        print(f"ERROR MSG : {e}", flush=True)
        print("\nTRACEBACK:", flush=True)
        traceback.print_exc()
        raise

    finally:
        if env is not None:
            try:
                env.close()
                print("DEBUG: env.close() done", flush=True)
            except Exception as close_err:
                print(f"WARNING: env.close() failed: {close_err}", flush=True)

        if success:
            print("Training complete!", flush=True)
        else:
            print("Training aborted due to an error.", flush=True)


if __name__ == "__main__":
    main()