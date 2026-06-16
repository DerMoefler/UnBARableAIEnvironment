"""
Training script integrating BAR environment, policy, replay buffer, and R_MAPPO trainer.

Berücksichtigt BAR-/MAPPO-spezifische Konzepte aus dem BARRunner:
- active_masks
- bad_masks
- available_actions
- rnn_states / rnn_states_critic
- per-agent done handling
- robustes Buffer-Insert (kurze oder lange Signatur)

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
import inspect
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


def _prepare_share_obs(
    share_obs: Any,
    obs: np.ndarray,
    num_agents: int,
    obs_dim: int,
    use_centralized_v: bool = True,
) -> np.ndarray:
    """
    Bereitet share_obs robust vor.

    Fälle:
    - share_obs ist None -> aus obs ableiten
    - use_centralized_v=False -> share_obs = obs
    - use_centralized_v=True, aber Env liefert nichts -> globaler Mittelwert
    """
    if not use_centralized_v:
        return obs.astype(np.float32)

    if share_obs is None:
        # globaler Zustand aus mean über Agenten
        return obs.mean(axis=0).astype(np.float32)

    arr = np.asarray(share_obs, dtype=np.float32)

    # Falls globaler State bereits 1D ist -> ok
    if arr.ndim == 1:
        if arr.size == obs_dim:
            return arr.astype(np.float32)
        fixed = np.zeros((obs_dim,), dtype=np.float32)
        n = min(obs_dim, arr.size)
        fixed[:n] = arr.reshape(-1)[:n]
        return fixed

    # Falls per-agent share_obs geliefert wird
    if arr.ndim >= 2:
        if arr.shape[0] == num_agents:
            arr = arr.reshape(num_agents, -1)
            if arr.shape[1] == obs_dim:
                # globalen zentralen state als mean daraus bilden
                return arr.mean(axis=0).astype(np.float32)

            fixed = np.zeros((num_agents, obs_dim), dtype=np.float32)
            n = min(obs_dim, arr.shape[1])
            fixed[:, :n] = arr[:, :n]
            return fixed.mean(axis=0).astype(np.float32)

        # Sonst flatten + pad
        flat = arr.reshape(-1)
        fixed = np.zeros((obs_dim,), dtype=np.float32)
        n = min(obs_dim, flat.size)
        fixed[:n] = flat[:n]
        return fixed

    return obs.mean(axis=0).astype(np.float32)


def _prepare_available_actions(
    available_actions: Any,
    num_agents: int,
    action_dim: int,
) -> np.ndarray:
    """
    Bringt available_actions robust in Shape: (num_agents, action_dim)

    Falls Env keine legalen Aktionen liefert:
    -> alles erlaubt (1en)
    """
    if available_actions is None:
        return np.ones((num_agents, action_dim), dtype=np.float32)

    arr = np.asarray(available_actions, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size == action_dim:
            arr = np.tile(arr[None, :], (num_agents, 1))
        elif arr.size == num_agents * action_dim:
            arr = arr.reshape(num_agents, action_dim)
        else:
            fixed = np.ones((num_agents * action_dim,), dtype=np.float32)
            n = min(fixed.size, arr.size)
            fixed[:n] = arr.reshape(-1)[:n]
            arr = fixed.reshape(num_agents, action_dim)

    elif arr.ndim >= 2:
        if arr.shape[0] != num_agents:
            flat = arr.reshape(-1)
            fixed = np.ones((num_agents * action_dim,), dtype=np.float32)
            n = min(fixed.size, flat.size)
            fixed[:n] = flat[:n]
            arr = fixed.reshape(num_agents, action_dim)
        else:
            arr = arr.reshape(num_agents, -1)
            if arr.shape[1] != action_dim:
                fixed = np.ones((num_agents, action_dim), dtype=np.float32)
                n = min(action_dim, arr.shape[1])
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


def _prepare_dones(done_like: Any, num_agents: int) -> np.ndarray:
    """
    Bringt einzelne Agent-dones in Shape: (num_agents,)
    """
    if done_like is None:
        return np.zeros((num_agents,), dtype=bool)

    arr = np.asarray(done_like, dtype=bool)

    if arr.ndim == 0:
        return np.full((num_agents,), bool(arr), dtype=bool)

    arr = arr.reshape(-1)
    fixed = np.zeros((num_agents,), dtype=bool)
    n = min(num_agents, arr.size)
    fixed[:n] = arr[:n]
    return fixed


def _repeat_value(value_tensor: torch.Tensor, num_agents: int) -> np.ndarray:
    """
    Macht aus Critic-Output einen Buffer-kompatiblen Array.
    Zielshape: (num_agents, 1)
    """
    value_np = value_tensor.detach().cpu().numpy().reshape(-1)

    if value_np.size == 1:
        return np.full((num_agents, 1), float(value_np[0]), dtype=np.float32)

    fixed = np.zeros((num_agents,), dtype=np.float32)
    n = min(num_agents, value_np.size)
    fixed[:n] = value_np[:n]
    return fixed.reshape(num_agents, 1)


def _extract_bad_masks(step_info: Any, num_agents: int) -> np.ndarray:
    """
    bad_masks:
    0.0 -> bad_transition=True (z.B. TimeLimit)
    1.0 -> normaler Übergang

    Erwartet BAR-ähnliche info-Strukturen, z.B.:
    - dict pro Agent
    - Liste/Tuple von Agent-Infos
    """
    bad_masks = np.ones((num_agents, 1), dtype=np.float32)

    if step_info is None:
        return bad_masks

    # Fall: step_info ist dict
    if isinstance(step_info, dict):
        # globales bad_transition
        if "bad_transition" in step_info:
            val = 0.0 if bool(step_info["bad_transition"]) else 1.0
            return np.full((num_agents, 1), val, dtype=np.float32)

        # agent_i -> dict
        for i in range(num_agents):
            agent_key = i
            if agent_key in step_info and isinstance(step_info[agent_key], dict):
                val = 0.0 if bool(step_info[agent_key].get("bad_transition", False)) else 1.0
                bad_masks[i, 0] = val

        return bad_masks

    # Fall: Liste / Tuple pro Agent
    if isinstance(step_info, (list, tuple)):
        for i in range(min(num_agents, len(step_info))):
            item = step_info[i]
            if isinstance(item, dict):
                bad_masks[i, 0] = 0.0 if bool(item.get("bad_transition", False)) else 1.0
        return bad_masks

    return bad_masks


def _infer_buffer_insert_mode(buffer) -> str:
    """
    Erkennt grob, welche insert-Signatur der Buffer hat.

    Rückgabe:
    - "extended" -> MAPPO-style insert(...)
    - "simple"   -> vereinfachte Signatur
    """
    try:
        sig = inspect.signature(buffer.insert)
        params = list(sig.parameters.keys())

        # MAPPO-artig
        if "rnn_states" in params or len(params) >= 10:
            return "extended"

        return "simple"
    except Exception:
        return "simple"


def _maybe_buffer_has_available_actions(buffer) -> bool:
    return hasattr(buffer, "available_actions")


def _make_rnn_state_arrays(
    num_agents: int,
    recurrent_n: int = 1,
    hidden_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Erzeugt Default-RNN-Zustände für nicht/unklar rekurrente Policies.
    """
    rnn_shape = (num_agents, recurrent_n, hidden_size)
    return (
        np.zeros(rnn_shape, dtype=np.float32),
        np.zeros(rnn_shape, dtype=np.float32),
    )


def _policy_sample_actions(
    policy,
    obs: np.ndarray,
    share_obs: np.ndarray,
    available_actions: np.ndarray,
    device: torch.device,
    num_agents: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robustes Aktionssampling:
    1) Wenn policy.get_actions(...) existiert, nutze es
    2) Sonst fallback auf actor/critic direkt
    """
    # ---------------------------------------------------------
    # bevorzugt: echte MAPPO-Policy API
    # ---------------------------------------------------------
    if hasattr(policy, "get_actions") and callable(policy.get_actions):
        # Dummy RNN-States/Masks für nicht-rekurrenten Fall
        rnn_states, rnn_states_critic = _make_rnn_state_arrays(num_agents)
        masks = np.ones((num_agents, 1), dtype=np.float32)

        value, action, action_log_prob, _, _ = policy.get_actions(
            share_obs if share_obs.ndim > 1 else np.repeat(share_obs[None, :], num_agents, axis=0),
            obs,
            rnn_states,
            rnn_states_critic,
            masks,
            available_actions,
        )

        value_np = value.detach().cpu().numpy().reshape(-1)
        action_np = action.detach().cpu().numpy().reshape(num_agents, 1)
        action_log_prob_np = action_log_prob.detach().cpu().numpy().reshape(num_agents, 1)

        if value_np.size == 1:
            values = np.full((num_agents, 1), float(value_np[0]), dtype=np.float32)
        else:
            fixed = np.zeros((num_agents,), dtype=np.float32)
            n = min(num_agents, value_np.size)
            fixed[:n] = value_np[:n]
            values = fixed.reshape(num_agents, 1)

        return values, action_np, action_log_prob_np

    # ---------------------------------------------------------
    # fallback: direkter actor/critic Zugriff
    # ---------------------------------------------------------
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

    with torch.no_grad():
        action_logits = policy.actor(obs_tensor)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions = action_dist.sample()
        action_log_probs = action_dist.log_prob(actions)

        share_obs_tensor = torch.as_tensor(
            share_obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        values = policy.critic(share_obs_tensor)

    value_preds = _repeat_value(values, num_agents)
    action_np = actions.detach().cpu().numpy().reshape(num_agents, 1)
    action_log_prob_np = action_log_probs.detach().cpu().numpy().reshape(num_agents, 1)

    return value_preds, action_np, action_log_prob_np


def _reset_rnn_states_for_done_env(
    rnn_states: np.ndarray,
    rnn_states_critic: np.ndarray,
    done_env: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bei Episode-Ende RNN-Stati nullen.
    """
    if done_env:
        rnn_states[:] = 0.0
        rnn_states_critic[:] = 0.0
    return rnn_states, rnn_states_critic


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
    parser.add_argument(
        "--use-centralized-v",
        action="store_true",
        default=True,
        help="Use centralized critic input (share_obs).",
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

        insert_mode = _infer_buffer_insert_mode(buffer)
        print(f"DEBUG: buffer insert mode = {insert_mode}", flush=True)

        print("Initializing R_MAPPO trainer...", flush=True)
        trainer_args = TrainerArgs()
        trainer_args.num_mini_batch = args.num_mini_batch
        trainer = R_MAPPO(trainer_args, policy, device=device)
        print("DEBUG: trainer created", flush=True)

        print("\nStarting training loop...", flush=True)

        # Default nicht-rekurrente states
        recurrent_n = 1
        hidden_size = 1

        for episode in range(args.num_episodes):
            print(f"DEBUG: starting episode {episode + 1}", flush=True)

            # -------------------------------------------------------------
            # Reset
            # -------------------------------------------------------------
            reset_result = env.reset()
            print("DEBUG: env.reset() returned", flush=True)

            obs_raw = None
            share_obs_raw = None
            info = {}
            available_actions_raw = None

            if isinstance(reset_result, tuple):
                # mögliche Fälle:
                # (obs, info)
                # (obs, share_obs, available_actions)
                # (obs, share_obs, info, available_actions)
                if len(reset_result) >= 1:
                    obs_raw = reset_result[0]
                if len(reset_result) >= 2:
                    second = reset_result[1]
                    if isinstance(second, dict):
                        info = second
                    else:
                        share_obs_raw = second
                if len(reset_result) >= 3:
                    third = reset_result[2]
                    if isinstance(third, dict):
                        info = third
                    else:
                        available_actions_raw = third
                if len(reset_result) >= 4:
                    fourth = reset_result[3]
                    if isinstance(fourth, dict):
                        info = fourth
                    else:
                        available_actions_raw = fourth
            else:
                obs_raw = reset_result

            obs = _prepare_obs(obs_raw, args.num_agents, obs_dim)
            share_obs = _prepare_share_obs(
                share_obs_raw,
                obs,
                args.num_agents,
                obs_dim,
                use_centralized_v=args.use_centralized_v,
            )
            available_actions = _prepare_available_actions(
                available_actions_raw,
                args.num_agents,
                action_dim,
            )

            if args.debug_shapes:
                print(f"DEBUG: reset obs shape = {obs.shape}", flush=True)
                print(f"DEBUG: reset share_obs shape = {share_obs.shape}", flush=True)
                print(f"DEBUG: reset available_actions shape = {available_actions.shape}", flush=True)

            episode_reward = 0.0
            done = False
            step_count = 0

            rnn_states, rnn_states_critic = _make_rnn_state_arrays(
                args.num_agents, recurrent_n=recurrent_n, hidden_size=hidden_size
            )

            trainer.prep_rollout()
            print("DEBUG: trainer.prep_rollout() done", flush=True)

            while not done and step_count < args.buffer_size:
                if args.debug_shapes:
                    print(f"DEBUG: rollout step {step_count}", flush=True)

                value_preds, actions, action_log_probs = _policy_sample_actions(
                    policy=policy,
                    obs=obs,
                    share_obs=share_obs,
                    available_actions=available_actions,
                    device=device,
                    num_agents=args.num_agents,
                )

                env_actions = actions.reshape(args.num_agents)

                if args.debug_shapes:
                    print(f"DEBUG: env_actions shape = {env_actions.shape}", flush=True)

                # ---------------------------------------------------------
                # Environment step
                # ---------------------------------------------------------
                step_result = env.step(env_actions)

                # 기대하는 형태:
                # A) (obs, reward, terminated, truncated, info)
                # B) (obs, share_obs, reward, terminated, truncated, info)
                # C) (obs, share_obs, reward, terminated, truncated, info, available_actions)
                if not isinstance(step_result, tuple):
                    raise RuntimeError(
                        "env.step(...) muss ein Tupel liefern."
                    )

                next_obs_raw = None
                next_share_obs_raw = None
                reward = None
                terminated = None
                truncated = None
                step_info = {}
                next_available_actions_raw = None

                if len(step_result) == 5:
                    next_obs_raw, reward, terminated, truncated, step_info = step_result
                elif len(step_result) == 6:
                    next_obs_raw, next_share_obs_raw, reward, terminated, truncated, step_info = step_result
                elif len(step_result) >= 7:
                    (
                        next_obs_raw,
                        next_share_obs_raw,
                        reward,
                        terminated,
                        truncated,
                        step_info,
                        next_available_actions_raw,
                    ) = step_result[:7]
                else:
                    raise RuntimeError(
                        "env.step(...) muss eines dieser Tupel liefern:\n"
                        "(obs, reward, terminated, truncated, info)\n"
                        "(obs, share_obs, reward, terminated, truncated, info)\n"
                        "(obs, share_obs, reward, terminated, truncated, info, available_actions)"
                    )

                # ---------------------------------------------------------
                # done-Handling analog BARRunner
                # ---------------------------------------------------------
                terminated_arr = _prepare_dones(terminated, args.num_agents)
                truncated_arr = _prepare_dones(truncated, args.num_agents)
                dones = np.logical_or(terminated_arr, truncated_arr)  # per-agent

                # Episode fertig? (wie im BARRunner: np.all(dones, axis=...))
                done_env = bool(np.all(dones))
                done = done_env

                next_obs = _prepare_obs(next_obs_raw, args.num_agents, obs_dim)
                next_share_obs = _prepare_share_obs(
                    next_share_obs_raw,
                    next_obs,
                    args.num_agents,
                    obs_dim,
                    use_centralized_v=args.use_centralized_v,
                )
                next_available_actions = _prepare_available_actions(
                    next_available_actions_raw,
                    args.num_agents,
                    action_dim,
                )

                reward_arr = _prepare_reward(reward, args.num_agents)

                # masks: 0 wenn Episode komplett fertig, sonst 1
                masks = np.ones((args.num_agents, 1), dtype=np.float32)
                if done_env:
                    masks[:] = 0.0

                # active_masks:
                # 0 für tote Agenten
                # aber bei Episode-Ende wieder alles auf 1 für reset-ähnliches Verhalten
                active_masks = np.ones((args.num_agents, 1), dtype=np.float32)
                active_masks[dones] = 0.0
                if done_env:
                    active_masks[:] = 1.0

                bad_masks = _extract_bad_masks(step_info, args.num_agents)

                # RNN states bei Episode-Ende zurücksetzen
                rnn_states, rnn_states_critic = _reset_rnn_states_for_done_env(
                    rnn_states, rnn_states_critic, done_env
                )

                if args.debug_shapes:
                    print(f"DEBUG: obs shape = {obs.shape}", flush=True)
                    print(f"DEBUG: share_obs shape = {share_obs.shape}", flush=True)
                    print(f"DEBUG: next_obs shape = {next_obs.shape}", flush=True)
                    print(f"DEBUG: next_share_obs shape = {next_share_obs.shape}", flush=True)
                    print(f"DEBUG: actions shape = {actions.shape}", flush=True)
                    print(f"DEBUG: action_log_probs shape = {action_log_probs.shape}", flush=True)
                    print(f"DEBUG: value_preds shape = {value_preds.shape}", flush=True)
                    print(f"DEBUG: reward_arr shape = {reward_arr.shape}", flush=True)
                    print(f"DEBUG: masks shape = {masks.shape}", flush=True)
                    print(f"DEBUG: active_masks shape = {active_masks.shape}", flush=True)
                    print(f"DEBUG: bad_masks shape = {bad_masks.shape}", flush=True)
                    print(f"DEBUG: available_actions shape = {available_actions.shape}", flush=True)

                # ---------------------------------------------------------
                # Buffer insert
                # ---------------------------------------------------------
                if insert_mode == "extended":
                    # MAPPO/BARRunner-artige Signatur
                    try:
                        buffer.insert(
                            share_obs,
                            obs,
                            rnn_states,
                            rnn_states_critic,
                            actions,
                            action_log_probs,
                            value_preds,
                            reward_arr,
                            masks,
                            bad_masks,
                            active_masks,
                            available_actions,
                        )
                    except TypeError:
                        # Falls keyword-basierte Signatur existiert
                        buffer.insert(
                            share_obs=share_obs,
                            obs=obs,
                            rnn_states=rnn_states,
                            rnn_states_critic=rnn_states_critic,
                            actions=actions,
                            action_log_probs=action_log_probs,
                            value_preds=value_preds,
                            rewards=reward_arr,
                            masks=masks,
                            bad_masks=bad_masks,
                            active_masks=active_masks,
                            available_actions=available_actions,
                        )
                else:
                    # vereinfachte Signatur
                    try:
                        buffer.insert(
                            share_obs=share_obs,
                            obs=obs,
                            actions=actions,
                            action_log_probs=action_log_probs,
                            value_preds=value_preds,
                            rewards=reward_arr,
                            masks=masks,
                            active_masks=active_masks,
                        )
                    except TypeError:
                        buffer.insert(
                            share_obs,
                            obs,
                            actions,
                            action_log_probs,
                            value_preds,
                            reward_arr,
                            masks,
                            active_masks,
                        )

                episode_reward += float(reward_arr.mean())
                step_count += 1

                obs = next_obs
                share_obs = next_share_obs
                available_actions = next_available_actions

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

            # compute_returns kompatibel aufrufen
            try:
                buffer.compute_returns(next_value, gamma=args.gamma)
            except TypeError:
                buffer.compute_returns(next_value)

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