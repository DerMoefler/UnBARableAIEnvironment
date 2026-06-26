"""
Simulated BAR 3v3 pawn training test.

Run:
    uv run tests/test_bar_3v3_training.py

This version logs:
- episode reward
- steps
- win/loss
- final ally alive
- final enemy alive
- enemy damage done
- own damage taken
- action counts
- action trace
- entropy
- policy/value loss
"""

from __future__ import annotations

import argparse
import inspect
import random
import traceback
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# IMPORTANT:
# If this import does not match your project, replace it with the import
# from your old tests/test_bar_3v3_training.py file.
# ---------------------------------------------------------------------------
try:
    from src.environment.simulated_bar_3v3_pawn_env import SimulatedBAR3v3PawnEnv
except ImportError:
    try:
        from src.environment.bar_3v3_pawn_env import SimulatedBAR3v3PawnEnv
    except ImportError:
        try:
            from src.environment.bar_environment import SimulatedBAR3v3PawnEnv
        except ImportError as import_err:
            raise ImportError(
                "Could not import SimulatedBAR3v3PawnEnv.\n"
                "Open your old tests/test_bar_3v3_training.py and copy the correct "
                "SimulatedBAR3v3PawnEnv import into this file."
            ) from import_err


from src.train.policy import R_MAPPO_Policy
from src.train.replay_buffer import SharedReplayBuffer
from src.train.r_mappo import R_MAPPO


# ---------------------------------------------------------------------------
# Trainer args
# ---------------------------------------------------------------------------
class TrainerArgs:
    def __init__(self) -> None:
        self.clip_param = 0.2
        self.ppo_epoch = 10
        self.num_mini_batch = 4
        self.data_chunk_length = 4
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.05
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_obs(obs: Any, num_agents: int, obs_dim: int) -> np.ndarray:
    if obs is None:
        return np.zeros((num_agents, obs_dim), dtype=np.float32)

    arr = np.asarray(obs, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size == obs_dim:
            arr = np.tile(arr[None, :], (num_agents, 1))
        elif arr.size == num_agents * obs_dim:
            arr = arr.reshape(num_agents, obs_dim)
        else:
            fixed = np.zeros((num_agents * obs_dim,), dtype=np.float32)
            n = min(fixed.size, arr.size)
            fixed[:n] = arr.reshape(-1)[:n]
            arr = fixed.reshape(num_agents, obs_dim)

    elif arr.ndim >= 2:
        if arr.shape[0] != num_agents:
            flat = arr.reshape(-1)
            fixed = np.zeros((num_agents * obs_dim,), dtype=np.float32)
            n = min(fixed.size, flat.size)
            fixed[:n] = flat[:n]
            arr = fixed.reshape(num_agents, obs_dim)
        else:
            arr = arr.reshape(num_agents, -1)

            if arr.shape[1] != obs_dim:
                fixed = np.zeros((num_agents, obs_dim), dtype=np.float32)
                n = min(obs_dim, arr.shape[1])
                fixed[:, :n] = arr[:, :n]
                arr = fixed

    return arr.astype(np.float32)


def _prepare_share_obs(
    share_obs: Any,
    obs: np.ndarray,
    num_agents: int,
    obs_dim: int,
) -> np.ndarray:
    if share_obs is None:
        return obs.mean(axis=0).astype(np.float32)

    arr = np.asarray(share_obs, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size == obs_dim:
            return arr.astype(np.float32)

        fixed = np.zeros((obs_dim,), dtype=np.float32)
        n = min(obs_dim, arr.size)
        fixed[:n] = arr.reshape(-1)[:n]
        return fixed

    if arr.ndim >= 2:
        if arr.shape[0] == num_agents:
            arr = arr.reshape(num_agents, -1)

            if arr.shape[1] == obs_dim:
                return arr.mean(axis=0).astype(np.float32)

            fixed = np.zeros((num_agents, obs_dim), dtype=np.float32)
            n = min(obs_dim, arr.shape[1])
            fixed[:, :n] = arr[:, :n]
            return fixed.mean(axis=0).astype(np.float32)

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
        arr = arr[:, :1]

    return arr.astype(np.float32)


def _prepare_dones(done_like: Any, num_agents: int) -> np.ndarray:
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
    value_np = value_tensor.detach().cpu().numpy().reshape(-1)

    if value_np.size == 1:
        return np.full((num_agents, 1), float(value_np[0]), dtype=np.float32)

    fixed = np.zeros((num_agents,), dtype=np.float32)
    n = min(num_agents, value_np.size)
    fixed[:n] = value_np[:n]

    return fixed.reshape(num_agents, 1)


def _extract_bad_masks(step_info: Any, num_agents: int) -> np.ndarray:
    bad_masks = np.ones((num_agents, 1), dtype=np.float32)

    if step_info is None:
        return bad_masks

    if isinstance(step_info, dict):
        if "bad_transition" in step_info:
            val = 0.0 if bool(step_info["bad_transition"]) else 1.0
            return np.full((num_agents, 1), val, dtype=np.float32)

        for i in range(num_agents):
            if i in step_info and isinstance(step_info[i], dict):
                val = 0.0 if bool(step_info[i].get("bad_transition", False)) else 1.0
                bad_masks[i, 0] = val

        return bad_masks

    if isinstance(step_info, (list, tuple)):
        for i in range(min(num_agents, len(step_info))):
            item = step_info[i]

            if isinstance(item, dict):
                bad_masks[i, 0] = 0.0 if bool(item.get("bad_transition", False)) else 1.0

        return bad_masks

    return bad_masks


def _extract_episode_debug_from_info(step_info: Any) -> dict[str, Any]:
    """
    Extract win/loss/alive/debug metrics from env info.

    Supports:
    - dict
    - list/tuple of dicts
    """

    result: dict[str, Any] = {
        "won": False,
        "lost": False,
        "ally_alive": None,
        "enemy_alive": None,
        "enemy_damage_done": 0.0,
        "own_damage_taken": 0.0,
        "info_keys": [],
    }

    infos: list[dict[str, Any]] = []

    if isinstance(step_info, dict):
        infos = [step_info]

    elif isinstance(step_info, (list, tuple)):
        infos = [x for x in step_info if isinstance(x, dict)]

    for info in infos:
        result["info_keys"].extend(list(info.keys()))

        if bool(info.get("won", False)):
            result["won"] = True

        if bool(info.get("lost", False)):
            result["lost"] = True

        for ally_key in [
            "ally_alive",
            "allies_alive",
            "num_ally_alive",
            "final_ally_alive",
            "ally_units_alive",
            "alive_allies",
        ]:
            if ally_key in info:
                result["ally_alive"] = info[ally_key]

        for enemy_key in [
            "enemy_alive",
            "enemies_alive",
            "num_enemy_alive",
            "final_enemy_alive",
            "enemy_units_alive",
            "alive_enemies",
        ]:
            if enemy_key in info:
                result["enemy_alive"] = info[enemy_key]

        for damage_key in [
            "enemy_damage_done",
            "damage_done",
            "damage_to_enemy",
            "enemy_damage",
        ]:
            if damage_key in info:
                result["enemy_damage_done"] += float(info.get(damage_key, 0.0))
                break

        for damage_key in [
            "own_damage_taken",
            "damage_taken",
            "ally_damage_taken",
            "own_damage",
        ]:
            if damage_key in info:
                result["own_damage_taken"] += float(info.get(damage_key, 0.0))
                break

    # Infer win/loss if explicit won/lost is not provided.
    try:
        if result["enemy_alive"] is not None and int(result["enemy_alive"]) <= 0:
            result["won"] = True
    except Exception:
        pass

    try:
        if result["ally_alive"] is not None and int(result["ally_alive"]) <= 0:
            result["lost"] = True
    except Exception:
        pass

    result["info_keys"] = sorted(set(result["info_keys"]))

    return result


def _infer_buffer_insert_mode(buffer: Any) -> str:
    try:
        sig = inspect.signature(buffer.insert)
        params = list(sig.parameters.keys())

        if "rnn_states" in params or len(params) >= 10:
            return "extended"

        return "simple"

    except Exception:
        return "simple"


def _make_rnn_state_arrays(
    num_agents: int,
    recurrent_n: int = 1,
    hidden_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    rnn_shape = (num_agents, recurrent_n, hidden_size)

    return (
        np.zeros(rnn_shape, dtype=np.float32),
        np.zeros(rnn_shape, dtype=np.float32),
    )


def _policy_sample_actions(
    policy: Any,
    obs: np.ndarray,
    share_obs: np.ndarray,
    available_actions: np.ndarray,
    device: torch.device,
    num_agents: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(policy, "get_actions") and callable(policy.get_actions):
        rnn_states, rnn_states_critic = _make_rnn_state_arrays(num_agents)
        masks = np.ones((num_agents, 1), dtype=np.float32)

        share_obs_for_policy = (
            share_obs
            if share_obs.ndim > 1
            else np.repeat(share_obs[None, :], num_agents, axis=0)
        )

        value, action, action_log_prob, _, _ = policy.get_actions(
            share_obs_for_policy,
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

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

    with torch.no_grad():
        action_logits = policy.actor(obs_tensor)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions = action_dist.sample()
        action_log_probs = action_dist.log_prob(actions)

        share_obs_tensor = torch.as_tensor(
            share_obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        values = policy.critic(share_obs_tensor)

    value_preds = _repeat_value(values, num_agents)
    action_np = actions.detach().cpu().numpy().reshape(num_agents, 1)
    action_log_prob_np = action_log_probs.detach().cpu().numpy().reshape(num_agents, 1)

    return value_preds, action_np, action_log_prob_np


def _create_env(args: argparse.Namespace):
    """
    Creates the simulated env while tolerating different constructor signatures.
    """

    try:
        return SimulatedBAR3v3PawnEnv(
            max_steps=args.max_steps,
            seed=args.seed,
            debug=args.debug_env,
        )
    except TypeError:
        pass

    try:
        return SimulatedBAR3v3PawnEnv(
            max_steps=args.max_steps,
            seed=args.seed,
        )
    except TypeError:
        pass

    try:
        return SimulatedBAR3v3PawnEnv()
    except TypeError as err:
        raise RuntimeError(
            "Could not construct SimulatedBAR3v3PawnEnv. "
            "Check the constructor in your old test file."
        ) from err


def _parse_reset_result(reset_result: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    obs_raw = None
    share_obs_raw = None
    available_actions_raw = None
    info: dict[str, Any] = {}

    if isinstance(reset_result, tuple):
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

    return obs_raw, share_obs_raw, available_actions_raw, info


def _parse_step_result(step_result: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    if not isinstance(step_result, tuple):
        raise RuntimeError("env.step(...) must return a tuple.")

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
        (
            next_obs_raw,
            next_share_obs_raw,
            reward,
            terminated,
            truncated,
            step_info,
        ) = step_result

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
            "env.step(...) must return one of:\n"
            "(obs, reward, terminated, truncated, info)\n"
            "(obs, share_obs, reward, terminated, truncated, info)\n"
            "(obs, share_obs, reward, terminated, truncated, info, available_actions)"
        )

    return (
        next_obs_raw,
        next_share_obs_raw,
        reward,
        terminated,
        truncated,
        step_info,
        next_available_actions_raw,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train simulated BAR 3v3 pawn environment with R_MAPPO"
    )

    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--num-mini-batch", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--obs-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-env", action="store_true")
    parser.add_argument("--debug-shapes", action="store_true")

    args = parser.parse_args()

    print(f"DEBUG: args = {args}", flush=True)

    _set_seed(args.seed)

    device = torch.device(args.device)
    print(f"DEBUG: device = {device}", flush=True)

    env = None
    success = False

    try:
        print("Initializing simulated BAR 3v3 pawn environment (no shared memory)...", flush=True)

        env = _create_env(args)

        print("DEBUG: SimulatedBAR3v3PawnEnv created", flush=True)

        obs_dim = args.obs_dim
        action_dim = args.action_dim

        print("Initializing policy...", flush=True)

        policy = R_MAPPO_Policy(
            obs_dim,
            action_dim,
            device=device,
            lr=args.lr,
        )

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

        trainer = R_MAPPO(
            trainer_args,
            policy,
            device=device,
        )

        print("DEBUG: trainer created", flush=True)
        print("\nStarting simulated 3v3 pawn training loop...", flush=True)

        recurrent_n = 1
        hidden_size = 1

        for episode in range(args.num_episodes):
            print(f"DEBUG: starting episode {episode + 1}", flush=True)

            reset_result = env.reset()

            print("DEBUG: env.reset() returned", flush=True)

            obs_raw, share_obs_raw, available_actions_raw, reset_info = _parse_reset_result(
                reset_result
            )

            print(f"DEBUG: initial env info = {reset_info}", flush=True)

            obs = _prepare_obs(
                obs_raw,
                args.num_agents,
                obs_dim,
            )

            share_obs = _prepare_share_obs(
                share_obs_raw,
                obs,
                args.num_agents,
                obs_dim,
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

            # -------------------------------------------------------------
            # Debug logging: actions, winner, alive counts
            # -------------------------------------------------------------
            action_counts = np.zeros(args.action_dim, dtype=np.int64)
            action_trace: list[list[int]] = []

            episode_won = False
            episode_lost = False
            final_ally_alive = None
            final_enemy_alive = None
            enemy_damage_done = 0.0
            own_damage_taken = 0.0
            last_info_keys: list[str] = []
            final_step_info: Any = None

            rnn_states, rnn_states_critic = _make_rnn_state_arrays(
                args.num_agents,
                recurrent_n=recurrent_n,
                hidden_size=hidden_size,
            )

            trainer.prep_rollout()

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

                # -------------------------------------------------------------
                # Log which actions were taken.
                # -------------------------------------------------------------
                env_actions_int = [int(a) for a in env_actions]
                action_trace.append(env_actions_int)

                for a in env_actions_int:
                    if 0 <= a < args.action_dim:
                        action_counts[a] += 1
                    else:
                        print(
                            f"WARNING: sampled invalid action {a}, "
                            f"expected range [0, {args.action_dim - 1}]",
                            flush=True,
                        )

                if args.debug_shapes:
                    print(f"DEBUG: env_actions = {env_actions_int}", flush=True)

                step_result = env.step(env_actions)

                (
                    next_obs_raw,
                    next_share_obs_raw,
                    reward,
                    terminated,
                    truncated,
                    step_info,
                    next_available_actions_raw,
                ) = _parse_step_result(step_result)

                final_step_info = step_info

                # -------------------------------------------------------------
                # Extract win/loss/alive/debug info from env info.
                # -------------------------------------------------------------
                debug_metrics = _extract_episode_debug_from_info(step_info)

                if debug_metrics["won"]:
                    episode_won = True

                if debug_metrics["lost"]:
                    episode_lost = True

                if debug_metrics["ally_alive"] is not None:
                    final_ally_alive = debug_metrics["ally_alive"]

                if debug_metrics["enemy_alive"] is not None:
                    final_enemy_alive = debug_metrics["enemy_alive"]

                enemy_damage_done += debug_metrics["enemy_damage_done"]
                own_damage_taken += debug_metrics["own_damage_taken"]

                if debug_metrics["info_keys"]:
                    last_info_keys = debug_metrics["info_keys"]

                terminated_arr = _prepare_dones(
                    terminated,
                    args.num_agents,
                )

                truncated_arr = _prepare_dones(
                    truncated,
                    args.num_agents,
                )

                dones = np.logical_or(
                    terminated_arr,
                    truncated_arr,
                )

                done_env = bool(np.all(dones))
                done = done_env

                next_obs = _prepare_obs(
                    next_obs_raw,
                    args.num_agents,
                    obs_dim,
                )

                next_share_obs = _prepare_share_obs(
                    next_share_obs_raw,
                    next_obs,
                    args.num_agents,
                    obs_dim,
                )

                next_available_actions = _prepare_available_actions(
                    next_available_actions_raw,
                    args.num_agents,
                    action_dim,
                )

                reward_arr = _prepare_reward(
                    reward,
                    args.num_agents,
                )

                masks = np.ones((args.num_agents, 1), dtype=np.float32)

                if done_env:
                    masks[:] = 0.0
                    rnn_states[:] = 0.0
                    rnn_states_critic[:] = 0.0

                active_masks = np.ones((args.num_agents, 1), dtype=np.float32)
                active_masks[dones] = 0.0

                if done_env:
                    active_masks[:] = 1.0

                bad_masks = _extract_bad_masks(
                    step_info,
                    args.num_agents,
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

                if insert_mode == "extended":
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
            # If env did not provide won/lost but alive counts exist, infer.
            # -------------------------------------------------------------
            try:
                if final_enemy_alive is not None and int(final_enemy_alive) <= 0:
                    episode_won = True
            except Exception:
                pass

            try:
                if final_ally_alive is not None and int(final_ally_alive) <= 0:
                    episode_lost = True
            except Exception:
                pass

            with torch.no_grad():
                share_obs_tensor = torch.as_tensor(
                    share_obs,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                next_value_tensor = policy.critic(share_obs_tensor)

            next_value = _repeat_value(
                next_value_tensor,
                args.num_agents,
            )

            try:
                buffer.compute_returns(
                    next_value,
                    gamma=args.gamma,
                )

            except TypeError:
                buffer.compute_returns(next_value)

            print("DEBUG: buffer.compute_returns() done", flush=True)

            trainer.prep_training()
            print("DEBUG: trainer.prep_training() done", flush=True)

            train_info = trainer.train(
                buffer,
                update_actor=True,
            )

            print("DEBUG: trainer.train() done", flush=True)

            buffer.reset()
            print("DEBUG: buffer.reset() done", flush=True)

            print(f"Episode {episode + 1}/{args.num_episodes}", flush=True)
            print(f"  Episode Reward: {episode_reward:.4f}", flush=True)
            print(f"  Steps: {step_count}", flush=True)

            print(f"  Won: {episode_won}", flush=True)
            print(f"  Lost: {episode_lost}", flush=True)
            print(f"  Final Ally Alive: {final_ally_alive}", flush=True)
            print(f"  Final Enemy Alive: {final_enemy_alive}", flush=True)
            print(f"  Enemy Damage Done: {enemy_damage_done:.4f}", flush=True)
            print(f"  Own Damage Taken: {own_damage_taken:.4f}", flush=True)

            print(f"  Action Counts: {action_counts.tolist()}", flush=True)
            print(f"  Action Trace: {action_trace}", flush=True)

            if last_info_keys:
                print(f"  Last Info Keys: {last_info_keys}", flush=True)
            else:
                print("  Last Info Keys: []", flush=True)

            if final_step_info is not None:
                print(f"  Final Step Info: {final_step_info}", flush=True)

            print(f"  Value Loss: {train_info.get('value_loss', 0):.6f}", flush=True)
            print(f"  Policy Loss: {train_info.get('policy_loss', 0):.6f}", flush=True)
            print(f"  Entropy: {train_info.get('dist_entropy', 0):.6f}", flush=True)
            print("", flush=True)

        success = True

    except Exception as e:
        print("\nERROR: Exception occurred during simulated 3v3 training!", flush=True)
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
            print("Simulated 3v3 pawn training complete!", flush=True)
        else:
            print("Simulated 3v3 pawn training aborted due to an error.", flush=True)


if __name__ == "__main__":
    main()