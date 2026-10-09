from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from contextlib import contextmanager
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import bar_ai

from src.environment.bar_environment import BAR_Environment
from src.train.policy import R_MAPPO_Policy
from src.train.replay_buffer import SharedReplayBuffer
from src.train.r_mappo import R_MAPPO
from src.train.obs_r_mappo import OBS_DIM, build_r_mappo_observations


# ---------------------------------------------------------------------------
# Trainer args
# ---------------------------------------------------------------------------
class TrainerArgs:
    """
    Default configuration values for the MAPPO training loop.

    Returns
    -------
    TrainerArgs
        Configuration object containing PPO and rollout settings.

    Examples
    --------
    >>> args = TrainerArgs()
    >>> args.clip_param
    0.2
    """

    def __init__(self) -> None:
        """
        Initializes the default MAPPO training configuration.

        Parameters
        ----------
        self : TrainerArgs
            The configuration instance.

        Returns
        -------
        None

        Examples
        --------
        >>> args = TrainerArgs()
        >>> args.num_mini_batch
        4
        """
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
    """
    Seeds the random number generators used by the training script.

    Parameters
    ----------
    seed : int
        Seed applied to Python, NumPy, and PyTorch random generators.

    Returns
    -------
    None

    Examples
    --------
    >>> _set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_bar_action(action_index: Any, unit_id: Any) -> bar_ai.Action:
    """Convert one policy action row into the object required by BAR_Environment.

    The policy samples zero-based action indices, but the bound C++ ActionId
    values are one-based. NumPy scalar values are converted to ordinary Python
    integers before crossing the pybind boundary.
    """
    action_value = int(action_index) + 1
    unit_value = int(unit_id)
    action_id = (
        bar_ai.ActionId(action_value)
        if hasattr(bar_ai, "ActionId")
        else action_value
    )

    # Prefer named construction, because it cannot silently swap two integer
    # constructor arguments.
    for kwargs in (
        {"action_id": action_id, "unit_id": unit_value},
        {"actionId": action_id, "unitId": unit_value},
    ):
        try:
            return bar_ai.Action(**kwargs)
        except TypeError:
            pass

    # Most pybind versions of Action expose Action(action_id, unit_id).
    try:
        return bar_ai.Action(action_id, unit_value)
    except TypeError:
        pass

    # Support a default-constructible binding with writable fields.
    try:
        result = bar_ai.Action()
    except TypeError as exc:
        raise TypeError(
            "Could not construct bar_ai.Action from the sampled action. "
            "Expected Action(action_id, unit_id), or a default constructor "
            "with writable action_id and unit_id fields."
        ) from exc

    action_field = next(
        (name for name in ("action_id", "actionId") if hasattr(result, name)),
        None,
    )
    unit_field = next(
        (name for name in ("unit_id", "unitId") if hasattr(result, name)),
        None,
    )
    if action_field is None or unit_field is None:
        raise TypeError(
            "bar_ai.Action is default-constructible, but its action and unit "
            "fields are not exposed to Python."
        )

    setattr(result, action_field, action_id)
    setattr(result, unit_field, unit_value)
    return result


def _step_all_agents(env: Any, actions: np.ndarray) -> Any:
    """Submit a policy action for each controlled unit.

    BAR_Environment.step accepts one bar_ai.Action, while R_MAPPO returns one
    [action_index, unit_id] row per agent. Each row is therefore converted and
    submitted separately. The final environment result is returned to the
    rollout loop, with scalar rewards accumulated over the submitted actions.
    """
    action_rows = np.asarray(actions)
    if action_rows.ndim == 1:
        action_rows = action_rows.reshape(1, -1)
    if action_rows.ndim != 2 or action_rows.shape[1] != 2:
        raise ValueError(
            "Policy actions must have shape (num_agents, 2), got "
            f"{action_rows.shape}."
        )

    final_result = None
    accumulated_reward = 0.0

    for action_index, unit_id in action_rows:
        result = env.step(_make_bar_action(action_index, unit_id))
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError(
                "BAR_Environment.step must return "
                "(obs, reward, terminated, truncated, info)."
            )

        obs, reward, terminated, truncated, info = result
        accumulated_reward += float(np.asarray(reward, dtype=np.float32).sum())
        final_result = (obs, accumulated_reward, terminated, truncated, info)

        # Do not submit further unit commands after the engine ended.
        if bool(np.all(np.asarray(terminated, dtype=bool))) or bool(
            np.all(np.asarray(truncated, dtype=bool))
        ):
            break

    if final_result is None:
        raise ValueError("At least one agent action is required.")
    return final_result


def _prepare_obs(
    obs: Any,
    num_agents: int,
    obs_dim: int,
    training_team_id: int = 0,
) -> np.ndarray:
    """
    Converts raw observations to a fixed per-agent float32 array.

    Parameters
    ----------
    obs : Any
        Raw observation data returned by the environment.
    num_agents : int
        Number of agents represented in the result.
    obs_dim : int
        Width of each agent observation.

    Returns
    -------
    observations : np.ndarray
        Array with shape `(num_agents, obs_dim)` and dtype float32.

    Examples
    --------
    >>> _prepare_obs(np.zeros(4), 2, 4).shape
    (2, 4)
    """
    if obs is None:
        return np.zeros((num_agents, obs_dim), dtype=np.float32)

    if isinstance(obs, dict):
        obs = build_r_mappo_observations(
            obs,
            training_team_id=training_team_id,
        )

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
    """
    Converts raw centralized observations to a fixed float32 vector.

    Parameters
    ----------
    share_obs : Any
        Raw shared observation data returned by the environment.
    obs : np.ndarray
        Prepared per-agent observations used as a fallback.
    num_agents : int
        Number of agents represented in the shared observation.
    obs_dim : int
        Width of the shared observation vector.

    Returns
    -------
    shared_observation : np.ndarray
        Vector with shape `(obs_dim,)` and dtype float32.

    Examples
    --------
    >>> obs = np.zeros((2, 4), dtype=np.float32)
    >>> _prepare_share_obs(None, obs, 2, 4).shape
    (4,)
    """
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
    """
    Converts raw legal-action data to a fixed per-agent mask array.

    Parameters
    ----------
    available_actions : Any
        Raw legal-action mask data returned by the environment.
    num_agents : int
        Number of agents represented in the result.
    action_dim : int
        Number of discrete actions.

    Returns
    -------
    action_mask : np.ndarray
        Array with shape `(num_agents, action_dim)` and dtype float32.

    Examples
    --------
    >>> _prepare_available_actions(None, 2, 3).shape
    (2, 3)
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
    Converts raw rewards to one scalar float32 reward per agent.

    Parameters
    ----------
    reward : Any
        Scalar or array-like reward data returned by the environment.
    num_agents : int
        Number of agents represented in the result.

    Returns
    -------
    rewards : np.ndarray
        Array with shape `(num_agents, 1)` and dtype float32.

    Examples
    --------
    >>> _prepare_reward(1.0, 2).shape
    (2, 1)
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
        arr = arr[:, :1]

    return arr.astype(np.float32)


def _prepare_dones(done_like: Any, num_agents: int) -> np.ndarray:
    """
    Converts termination data to one boolean flag per agent.

    Parameters
    ----------
    done_like : Any
        Scalar or array-like termination data.
    num_agents : int
        Number of agents represented in the result.

    Returns
    -------
    done_flags : np.ndarray
        Boolean array with shape `(num_agents,)`.

    Examples
    --------
    >>> _prepare_dones(True, 2).tolist()
    [True, True]
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
    Broadcasts critic values to one value per agent.

    Parameters
    ----------
    value_tensor : torch.Tensor
        Scalar or batched critic output.
    num_agents : int
        Number of agents represented in the result.

    Returns
    -------
    values : np.ndarray
        Array with shape `(num_agents, 1)` and dtype float32.

    Examples
    --------
    >>> _repeat_value(torch.tensor([[2.0]]), 3).shape
    (3, 1)
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
    Extracts bad-transition masks from environment step information.

    Parameters
    ----------
    step_info : Any
        Environment info dictionary or per-agent sequence of dictionaries.
    num_agents : int
        Number of agents represented in the result.

    Returns
    -------
    bad_masks : np.ndarray
        Float32 array with shape `(num_agents, 1)`, where zero marks a bad
        transition and one marks a normal transition.

    Examples
    --------
    >>> _extract_bad_masks({"bad_transition": True}, 2).tolist()
    [[0.0], [0.0]]
    """
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
    """
    Determines whether a replay buffer uses the extended insert signature.

    Parameters
    ----------
    buffer : Any
        Replay buffer exposing an `insert` method.

    Returns
    -------
    mode : str
        Either `"extended"` or `"simple"`.

    Examples
    --------
    >>> class Buffer:
    ...     def insert(self, a, b, c, d, e, f, g, h, i, rnn_states):
    ...         pass
    >>> _infer_buffer_insert_mode(Buffer())
    'extended'
    """
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
    """
    Allocates zero-filled actor and critic recurrent-state arrays.

    Parameters
    ----------
    num_agents : int
        Number of agents represented in each state array.
    recurrent_n : int, optional
        Number of recurrent layers, by default 1.
    hidden_size : int, optional
        Recurrent hidden-state width, by default 1.

    Returns
    -------
    states : tuple of np.ndarray
        Actor and critic state arrays with shape
        `(num_agents, recurrent_n, hidden_size)`.

    Examples
    --------
    >>> actor_states, critic_states = _make_rnn_state_arrays(2)
    >>> actor_states.shape
    (2, 1, 1)
    """
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
    """
    Samples actions and value predictions from a compatible policy interface.

    Parameters
    ----------
    policy : Any
        Policy exposing `get_actions`, `get_action`, or actor and critic modules.
    obs : np.ndarray
        Per-agent observations.
    share_obs : np.ndarray
        Shared observation for the critic.
    available_actions : np.ndarray
        Per-agent legal-action mask.
    device : torch.device
        Device used for tensor inference.
    num_agents : int
        Number of agents represented in the result.

    Returns
    -------
    values : np.ndarray
        Critic predictions with shape `(num_agents, 1)`.
    actions : np.ndarray
        Sampled actions with one row per agent.
    action_log_probs : np.ndarray
        Sampled action log-probabilities with shape `(num_agents, 1)`.

    Examples
    --------
    >>> class Policy:
    ...     pass
    >>> policy = Policy()
    >>> policy.actor = torch.nn.Linear(2, 3)
    >>> policy.critic = torch.nn.Linear(2, 1)
    >>> result = _policy_sample_actions(policy, np.zeros((2, 2)), np.zeros(2), np.ones((2, 3)), torch.device("cpu"), 2)
    >>> result[0].shape
    (2, 1)
    """
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
        action_np = action.detach().cpu().numpy().reshape(num_agents, 2)
        action_log_prob_np = action_log_prob.detach().cpu().numpy().reshape(num_agents, 1)

        if value_np.size == 1:
            values = np.full((num_agents, 1), float(value_np[0]), dtype=np.float32)
        else:
            fixed = np.zeros((num_agents,), dtype=np.float32)
            n = min(num_agents, value_np.size)
            fixed[:n] = value_np[:n]
            values = fixed.reshape(num_agents, 1)

        return values, action_np, action_log_prob_np

    if hasattr(policy, "get_action") and callable(policy.get_action):
        actions, action_log_probs = policy.get_action(
            torch.as_tensor(obs, dtype=torch.float32, device=device)
        )
        action_np = np.asarray(actions, dtype=np.float32).reshape(num_agents, 2)
        action_log_prob_np = np.asarray(action_log_probs, dtype=np.float32).reshape(num_agents, 1)

        with torch.no_grad():
            share_obs_tensor = torch.as_tensor(
                share_obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            values = policy.critic(share_obs_tensor)

        return _repeat_value(values, num_agents), action_np, action_log_prob_np

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


def _parse_reset_result(reset_result: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    """
    Normalizes supported environment reset return formats.

    Parameters
    ----------
    reset_result : Any
        Raw value returned by `env.reset()`.

    Returns
    -------
    parsed : tuple
        Observation, shared observation, available actions, and info values.

    Examples
    --------
    >>> _parse_reset_result(("obs", {"ready": True}))
    ('obs', None, None, {'ready': True})
    """
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
    """
    Normalizes supported environment step return formats.

    Parameters
    ----------
    step_result : tuple
        Raw tuple returned by `env.step()`.

    Returns
    -------
    parsed : tuple
        Next observation, shared observation, reward, termination flags, info,
        and available-action data.

    Raises
    ------
    RuntimeError
        If the environment does not return a supported tuple format.

    Examples
    --------
    >>> _parse_step_result(("obs", 1.0, False, False, {}))[0]
    'obs'
    """
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
@contextmanager
def _redirect_native_output(log_path: Path):
    """Redirect process-level stdout/stderr during native extension calls.

    contextlib.redirect_stdout is insufficient for C++ std::cout. This uses
    dup2, so output written by pybind/C++ code to file descriptors 1 and 2 is
    captured as well. Keep the context narrow because it is process-global.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with log_path.open("a", buffering=1) as log_file:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _format_duration(seconds: float) -> str:
    minutes, seconds = divmod(max(0.0, seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:06.3f}s"
    if minutes:
        return f"{minutes:d}m {seconds:06.3f}s"
    return f"{seconds:.3f}s"


def _warm_up_policy(
    policy: Any,
    device: torch.device,
    num_agents: int,
    obs_dim: int,
    action_dim: int,
) -> None:
    """Initialize lazy CPU/CUDA work before starting the real-time engine.

    The fast BAR match can finish while the first CUDA operation is still
    initializing its context and kernels. Running one synthetic inference
    before env.reset() removes that one-time delay from the live episode.
    """
    dummy_obs = np.zeros((num_agents, obs_dim), dtype=np.float32)
    dummy_share_obs = np.zeros((obs_dim,), dtype=np.float32)
    dummy_available_actions = np.ones(
        (num_agents, action_dim), dtype=np.float32
    )

    policy.actor.eval()
    policy.critic.eval()
    with torch.no_grad():
        _policy_sample_actions(
            policy=policy,
            obs=dummy_obs,
            share_obs=dummy_share_obs,
            available_actions=dummy_available_actions,
            device=device,
            num_agents=num_agents,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    policy.actor.train()
    policy.critic.train()


def _save_checkpoint(policy: Any, args: argparse.Namespace, episode: int, model_dir: Path) -> Path:
    """Save actor, critic, and reconstruction metadata atomically."""
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "episode": int(episode),
        "obs_dim": int(args.obs_dim),
        "action_dim": int(args.action_dim),
        "num_agents": int(args.num_agents),
        "training_team_id": int(args.training_team_id),
        "actor_state_dict": policy.actor.state_dict(),
        "critic_state_dict": policy.critic.state_dict(),
        "actor_optimizer_state_dict": (
            policy.actor_optimizer.state_dict()
            if hasattr(policy, "actor_optimizer") else None
        ),
        "critic_optimizer_state_dict": (
            policy.critic_optimizer.state_dict()
            if hasattr(policy, "critic_optimizer") else None
        ),
    }
    episode_path = model_dir / f"r_mappo_episode_{episode:04d}.pt"
    temporary_path = episode_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(episode_path)

    latest_path = model_dir / "r_mappo_latest.pt"
    latest_temporary_path = latest_path.with_suffix(".tmp")
    torch.save(checkpoint, latest_temporary_path)
    latest_temporary_path.replace(latest_path)
    return episode_path


def main() -> None:
    ###############
    #Parser
    ###############

    parser = argparse.ArgumentParser(
        description="Train simulated BAR 3v3 pawn environment with R_MAPPO"
    )

    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--num-mini-batch", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "cuda:0"))
    parser.add_argument("--obs-dim", type=int, default=OBS_DIM)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--training-team-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-env", action="store_true")
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models/r_mappo"),
        help="Parent directory containing named experiment runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Experiment name. Defaults to run_YYYY-MM-DD_HH-MM-SS.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint to load before continuing training.",
    )
    parser.add_argument(
        "--native-log",
        type=Path,
        default=None,
        help="Capture C/C++ stdout and stderr. Defaults to RUN_DIR/native.log.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save a numbered checkpoint every N episodes; 0 disables periodic saves.",
    )

    args = parser.parse_args()

    _set_seed(args.seed)

    ###############
    #Selecting Cuda
    ###############
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation cannot access it. "
            "Run the CUDA diagnostic shown in the documentation."
        )
    device_description = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" else "CPU"
    )

    ###############
    #Resuming
    ###############
    if args.resume is not None and args.run_name is None:
        run_dir = args.resume.resolve().parent
        args.run_name = run_dir.name
    else:
        args.run_name = args.run_name or datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        run_dir = args.model_root / args.run_name

    if run_dir.exists() and args.resume is None:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose another --run-name "
            "or pass --resume with a checkpoint from that run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG: run name = {args.run_name}", flush=True)
    print(f"DEBUG: run directory = {run_dir}", flush=True)
    native_log = args.native_log or (run_dir / "native.log")
    print(f"DEBUG: native output log = {native_log}", flush=True)
    run_started_at = time.perf_counter()
    total_engine_start_seconds = 0.0
    total_engine_step_seconds = 0.0
    total_inference_seconds = 0.0
    total_training_seconds = 0.0
    total_checkpoint_seconds = 0.0

    env = None
    success = False

    ###############
    #Starting the environment
    ###############

    try:
        # BAR_Environment cannot restart its gRPC server after an engine
        # session has ended. A fresh environment is therefore created for each
        # episode below instead of calling reset() on a closed environment.
        obs_dim = args.obs_dim
        action_dim = args.action_dim

        policy = R_MAPPO_Policy(
            obs_dim,
            action_dim,
            device=device,
            lr=args.lr,
        )

    ###############
    #resuming from checkpoint
    ###############
        
        start_episode = 0
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            if int(checkpoint["obs_dim"]) != obs_dim or int(checkpoint["action_dim"]) != action_dim:
                raise ValueError(
                    "Checkpoint dimensions do not match this run: "
                    f"checkpoint=({checkpoint['obs_dim']}, {checkpoint['action_dim']}), "
                    f"requested=({obs_dim}, {action_dim})."
                )
            policy.actor.load_state_dict(checkpoint["actor_state_dict"])
            policy.critic.load_state_dict(checkpoint["critic_state_dict"])
            if checkpoint.get("actor_optimizer_state_dict") is not None and hasattr(policy, "actor_optimizer"):
                policy.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            if checkpoint.get("critic_optimizer_state_dict") is not None and hasattr(policy, "critic_optimizer"):
                policy.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            start_episode = int(checkpoint.get("episode", 0))
            print(f"DEBUG: resumed from {args.resume} at episode {start_episode}", flush=True)

    ###############
    #creating replay buffer
    ###############
        buffer = SharedReplayBuffer(
            num_agents=args.num_agents,
            obs_shape=(obs_dim,),
            action_shape=(2,),
            buffer_size=args.buffer_size,
            device=device,
        )

    ###############
    #is buffer mode needed?
    ###############


        insert_mode = _infer_buffer_insert_mode(buffer)


    ###############
    #initializing trainer
    ###############
        trainer_args = TrainerArgs()
        trainer_args.num_mini_batch = args.num_mini_batch

        trainer = R_MAPPO(
            trainer_args,
            policy,
            device=device,
        )
    ###############
    #action mapping from nn to bar
    ###############
        original_action_mapper = trainer._map_policy_action_to_engine_action

        def map_policy_action_to_bar_action_id(policy_action_id):
            mapped_action_id = original_action_mapper(policy_action_id)
            if isinstance(mapped_action_id, bar_ai.ActionId):
                return mapped_action_id
            return bar_ai.ActionId(int(mapped_action_id))

        trainer._map_policy_action_to_engine_action = (
            map_policy_action_to_bar_action_id
        )

        print(f"Warming up policy on {device} before starting BAR...", flush=True)
        _warm_up_policy(
            policy=policy,
            device=device,
            num_agents=args.num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

    ###############
    #starting training loop
    ###############

        recurrent_n = 1
        hidden_size = 1

        for episode in range(start_episode, start_episode + args.num_episodes):
            print(f"DEBUG: starting episode {episode + 1}", flush=True)
            episode_started_at = time.perf_counter()
            inference_seconds = 0.0
            engine_start_seconds = 0.0
            engine_step_seconds = 0.0

            # Do not reuse an environment whose engine session or gRPC server
            # was stopped by the preceding episode.
            if env is not None:
                try:
                    env.close()
                    print("DEBUG: previous environment closed", flush=True)
                except Exception as close_err:
                    print(
                        f"WARNING: previous env.close() failed: {close_err}",
                        flush=True,
                    )
                finally:
                    env = None

            print("Initializing fresh BAR Environment...", flush=True)
            environment_started_at = time.perf_counter()
            with _redirect_native_output(native_log):
                env = BAR_Environment()
                reset_result = env.reset()
            engine_start_seconds += time.perf_counter() - environment_started_at
            total_engine_start_seconds += engine_start_seconds

            obs_raw, share_obs_raw, available_actions_raw, reset_info = _parse_reset_result(
                reset_result
            )

            obs = _prepare_obs(
                obs_raw,
                args.num_agents,
                obs_dim,
                training_team_id=args.training_team_id,
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

                inference_started_at = time.perf_counter()
                value_preds, actions, action_log_probs = _policy_sample_actions(
                    policy=policy,
                    obs=obs,
                    share_obs=share_obs,
                    available_actions=available_actions,
                    device=device,
                    num_agents=args.num_agents,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_elapsed = time.perf_counter() - inference_started_at
                inference_seconds += inference_elapsed
                total_inference_seconds += inference_elapsed

                env_actions = actions

                # -------------------------------------------------------------
                # Log which actions were taken.
                # -------------------------------------------------------------
                env_action_types = [int(a) for a in env_actions[:, 0]]
                action_trace.append(env_action_types)

                for a in env_action_types:
                    if 0 <= a < args.action_dim:
                        action_counts[a] += 1
                    else:
                        print(
                            f"WARNING: sampled invalid action {a}, "
                            f"expected range [0, {args.action_dim - 1}]",
                            flush=True,
                        )

                environment_started_at = time.perf_counter()
                with _redirect_native_output(native_log):
                    step_result = _step_all_agents(env, env_actions)
                engine_step_elapsed = time.perf_counter() - environment_started_at
                engine_step_seconds += engine_step_elapsed
                total_engine_step_seconds += engine_step_elapsed

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
                    ###############
                    #actually needed? Or does the reward calc do it?
                    ###############
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
                    training_team_id=args.training_team_id,
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

            ###############
            #Bruv wtf is buffer insert mode?
            ###############
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

            training_started_at = time.perf_counter()
            train_info = trainer.train(
                buffer,
                update_actor=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds = time.perf_counter() - training_started_at
            total_training_seconds += training_seconds

            print("DEBUG: trainer.train() done", flush=True)

            buffer.reset()
            print("DEBUG: buffer.reset() done", flush=True)

            print(f"Episode {episode + 1}/{start_episode + args.num_episodes}", flush=True)
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

            episode_seconds = time.perf_counter() - episode_started_at
            elapsed_seconds = time.perf_counter() - run_started_at
            completed_this_run = episode - start_episode + 1
            average_episode_seconds = elapsed_seconds / completed_this_run
            remaining_episodes = start_episode + args.num_episodes - episode - 1
            eta_seconds = average_episode_seconds * remaining_episodes
            print(f"  Timing: episode={_format_duration(episode_seconds)}", flush=True)
            print(f"          inference={_format_duration(inference_seconds)}", flush=True)
            print(f"          engine start/reset={_format_duration(engine_start_seconds)}", flush=True)
            print(f"          engine steps/waiting={_format_duration(engine_step_seconds)}", flush=True)
            print(f"          PPO update={_format_duration(training_seconds)}", flush=True)
            print(f"          elapsed={_format_duration(elapsed_seconds)}", flush=True)
            print(f"          ETA={_format_duration(eta_seconds)}", flush=True)

            if args.save_every > 0 and (episode + 1) % args.save_every == 0:
                checkpoint_started_at = time.perf_counter()
                checkpoint_path = _save_checkpoint(
                    policy, args, episode + 1, run_dir
                )
                total_checkpoint_seconds += time.perf_counter() - checkpoint_started_at
                print(f"  Saved checkpoint: {checkpoint_path}", flush=True)
            print("", flush=True)

        # Always save the final state, even when save_every is disabled or the
        # final episode is not an exact multiple of it.
        checkpoint_started_at = time.perf_counter()
        final_checkpoint = _save_checkpoint(
            policy, args, start_episode + args.num_episodes, run_dir
        )
        total_checkpoint_seconds += time.perf_counter() - checkpoint_started_at
        print(f"Final model saved to: {final_checkpoint}", flush=True)
        print(f"Latest model saved to: {run_dir / 'r_mappo_latest.pt'}", flush=True)

        total_runtime = time.perf_counter() - run_started_at
        measured_runtime = (
            total_engine_start_seconds
            + total_engine_step_seconds
            + total_inference_seconds
            + total_training_seconds
            + total_checkpoint_seconds
        )
        other_seconds = max(0.0, total_runtime - measured_runtime)

        def percentage(seconds: float) -> float:
            return 100.0 * seconds / total_runtime if total_runtime > 0.0 else 0.0

        print("\nOverall timing summary", flush=True)
        print(f"  Total runtime:          {_format_duration(total_runtime)} (100.00%)", flush=True)
        print(
            f"  Engine start/reset:     {_format_duration(total_engine_start_seconds)} "
            f"({percentage(total_engine_start_seconds):6.2f}%)",
            flush=True,
        )
        print(
            f"  Engine steps/waiting:   {_format_duration(total_engine_step_seconds)} "
            f"({percentage(total_engine_step_seconds):6.2f}%)",
            flush=True,
        )
        print(
            f"  Policy inference:       {_format_duration(total_inference_seconds)} "
            f"({percentage(total_inference_seconds):6.2f}%)",
            flush=True,
        )
        print(
            f"  PPO updates:            {_format_duration(total_training_seconds)} "
            f"({percentage(total_training_seconds):6.2f}%)",
            flush=True,
        )
        print(
            f"  Checkpoint writing:     {_format_duration(total_checkpoint_seconds)} "
            f"({percentage(total_checkpoint_seconds):6.2f}%)",
            flush=True,
        )
        print(
            f"  Other Python/logging:   {_format_duration(other_seconds)} "
            f"({percentage(other_seconds):6.2f}%)",
            flush=True,
        )
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
