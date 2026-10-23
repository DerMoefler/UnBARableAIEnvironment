import inspect
import json
import logging
import os
import random
import sys
import time
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
from src.train.mappo.r_mappo import R_MAPPO
from src.train.mappo.obs_r_mappo import OBS_DIM, build_r_mappo_observations
from src.train.mappo.trainerargs import TrainerArgs


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

    R_MAPPO returns one [action_index, unit_id] row per agent, while
    BAR_Environment.step accepts a list of bar_ai.Action objects. All agent
    actions are therefore converted and submitted together in one environment
    step.
    """
    action_rows = np.asarray(actions)
    if action_rows.ndim == 1:
        action_rows = action_rows.reshape(1, -1)
    if action_rows.ndim != 2 or action_rows.shape[1] != 2:
        raise ValueError(
            "Policy actions must have shape (num_agents, 2), got "
            f"{action_rows.shape}."
        )

    action_list = [
        _make_bar_action(action_index, unit_id)
        for action_index, unit_id in action_rows
    ]
    if not action_list:
        raise ValueError("At least one agent action is required.")

    result = env.step(action_list)
    if not isinstance(result, tuple) or len(result) != 5:
        raise RuntimeError(
            "BAR_Environment.step must return "
            "(obs, reward, terminated, truncated, info)."
        )
    return result


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
def _format_duration(seconds: float) -> str:
    minutes, seconds = divmod(max(0.0, seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:06.3f}s"
    if minutes:
        return f"{minutes:d}m {seconds:06.3f}s"
    return f"{seconds:.3f}s"


def _log_event(log_path: Path, log_level: int, event_level: int, event: str, **fields: Any) -> None:
    if event_level > log_level:
        return
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "level": event_level,
        "event": event,
        **fields,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, default=str, sort_keys=True) + "\n")
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[MAPPO:L{event_level}] {event}{': ' + details if details else ''}", flush=True)


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

#mappo specific
def _save_checkpoint(policy: Any, 
                     config: Any, 
                     episode: int, 
                     model_dir: Path) -> Path:
    """Save actor, critic, and reconstruction metadata atomically."""
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "episode": int(episode),
        "obs_dim": int(OBS_DIM),
        "action_dim": int(config["action_dimension"]),
        "num_agents": int(config["number_of_agents"]),
        "training_team_id": int(config["training_team_id"]),
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
    #import Json configfile
    with open("src/train/experiments/3v3Test/3v3Test.json", "r") as configfile:
        data = json.load(configfile)

    #start logging and configure log 
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logging.basicConfig(
        filename=f"src/train/experiments/{data["run_name"]}/{data["run_name"]}_{timestamp}.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        level=logging.INFO,
    )

    logging.info("Starting log")

    #Selecting Cuda
    if data["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logging.info("auto selection: cuda")
        else:
            logging.info("auto selection: CPU")
    else:
        device = torch.device(data["device"])
        logging.info("using configured device: %s", device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("cuda was selected, but cuda is unable to use")
    device_description = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" else "CPU"
    )
    logging.info("device_description: %s", device_description)

    #Choosing to resume another run or start a new one
    #Resuming not inplemented as of now.

    model_directory = Path(f"src/train/experiments/{data["run_name"]}/models")
    run_directory = Path(f"src/train/experiments/{data["run_name"]}/run")

    #Starting new run
    logging.info("run_name: %s", data["run_name"])
    logging.info("model_directory: %s", model_directory)
    run_started_at = time.perf_counter()
    total_engine_start_seconds = 0.0
    total_engine_step_seconds = 0.0
    total_inference_seconds = 0.0
    total_training_seconds = 0.0
    total_checkpoint_seconds = 0.0
    
    logging.info("Initializing new BAR Environment")
    env = BAR_Environment()
    logging.info("Initialized new BAR Environment")
    success = False

    try:
        obs_dim = OBS_DIM
        action_dim = data["action_dimension"]

        policy = R_MAPPO_Policy(
            obs_dim,
            action_dim,
            device=device,
            lr=data["learning_rate"],
        )
        logging.info("created R MAPPO policy")

        start_episode = 0

        #initializing Replay Buffer
        buffer = SharedReplayBuffer(
                    num_agents=data["number_of_agents"],
                    obs_shape=(obs_dim,),
                    action_shape=(2,),
                    buffer_size=data["buffer_size"],
                    device=device,
                )

        logging.info("created shared replay buffer")

        trainer_args = TrainerArgs()
        trainer_args.num_mini_batch = data["mini_batch"]

        trainer = R_MAPPO(
            trainer_args,
            policy,
            device=device,
        )

        #Warm up policy
        logging.info("Warming up policy on {device} before starting BAR")
        _warm_up_policy(
            policy=policy,
            device=device,
            num_agents=data["number_of_agents"],
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        recurrent_n = 1
        hidden_size = 1

        for episode in range(start_episode, start_episode + data["episodes"]):
            logging.debug("starting episode %d", episode + 1)
            engine_step_seconds = 0.0

            logging.info("starting environment reset")
            reset_result = env.reset()
            logging.info("reset complete")


            obs_raw, share_obs_raw, available_actions_raw, reset_info = _parse_reset_result(
                reset_result
            )

            obs = _prepare_obs(
                obs_raw,
                data["number_of_agents"],
                obs_dim,
                training_team_id=data["training_team_id"],
            )

            share_obs = _prepare_share_obs(
                share_obs_raw,
                obs,
                data["number_of_agents"],
                obs_dim,
            )

            available_actions = _prepare_available_actions(
                available_actions_raw,
                data["number_of_agents"],
                action_dim,
            )

            episode_reward = 0.0
            done = False
            step_count = 0

            action_counts = np.zeros(data["action_dimension"], dtype=np.int64)
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
                data["number_of_agents"],
                recurrent_n=recurrent_n,
                hidden_size=hidden_size,
            )

            trainer.prep_rollout()

            while not done and step_count < data["buffer_size"]:

                inference_started_at = time.perf_counter()
                value_preds, actions, action_log_probs = _policy_sample_actions(
                    policy=policy,
                    obs=obs,
                    share_obs=share_obs,
                    available_actions=available_actions,
                    device=device,
                    num_agents=data["number_of_agents"],
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                env_actions = actions

                #Logging taken actions
                env_action_types = [int(a) for a in env_actions[:, 0]]
                action_trace.append(env_action_types)

                for a in env_action_types:
                    if 0 <= a < data["action_dimension"]:
                        action_counts[a] += 1
                    else:
                        logging.warning("sampled invalid action %s expected range %s", a, [0, data["action_dimension"] - 1])

                environment_started_at = time.perf_counter()
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

                done_env = bool(terminated or truncated)

                dones = np.full(
                    data["number_of_agents"],
                    done_env,
                    dtype=bool,
                )
                done = done_env

                next_obs = _prepare_obs(
                    next_obs_raw,
                    data["number_of_agents"],
                    obs_dim,
                    training_team_id=data["training_team_id"],
                )

                next_share_obs = _prepare_share_obs(
                    next_share_obs_raw,
                    next_obs,
                    data["number_of_agents"],
                    obs_dim,
                )

                next_available_actions = _prepare_available_actions(
                    next_available_actions_raw,
                    data["number_of_agents"],
                    action_dim,
                )

                reward_arr = _prepare_reward(
                    reward,
                    data["number_of_agents"],
                )

                masks = np.ones((data["number_of_agents"], 1), dtype=np.float32)

                if done_env:
                    masks[:] = 0.0
                    rnn_states[:] = 0.0
                    rnn_states_critic[:] = 0.0

                active_masks = np.ones((data["number_of_agents"], 1), dtype=np.float32)
                active_masks[dones] = 0.0

                if done_env:
                    active_masks[:] = 1.0

                bad_masks = _extract_bad_masks(
                    step_info,
                    data["number_of_agents"],
                )

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
                data["number_of_agents"],
            )

            try:
                buffer.compute_returns(
                    next_value,
                    gamma=data["gamma"],
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


            logging.info("trainer.training() done")

            buffer.reset()
            logging.info("buffer.reset() done")

            logging.info("Episode %.2f",(episode + 1)/(start_episode + data["episodes"]))
            logging.info("Episode Reward: %.4f", episode_reward)
            logging.info("Steps: %d", step_count)

            logging.info("Won: %s", episode_won)
            logging.info("Lost: %s", episode_lost)
            logging.info("Final Ally Alive: %s",final_ally_alive)
            logging.info("Final Enemy Alive: %s",final_enemy_alive)
            logging.info("Enemy Damage Done: %.4f",enemy_damage_done)
            logging.info("Own Damage Taken: %.4f",own_damage_taken)

            logging.info("Action Counts: %s", action_counts.tolist())
            logging.info("Action Trace: %s",action_trace)

            if last_info_keys:
                logging.info("Last Info Keys: %s", last_info_keys)
            else:
                logging.info("Last Info Keys: []")

            if final_step_info is not None:
                logging.info("Final Step Info: %s", final_step_info)

            logging.info("Value Loss: %.6f", train_info.get('value_loss', 0))
            logging.info("Policy Loss: %.6f", train_info.get('policy_loss', 0))
            logging.info("Entropy: %.6f", train_info.get('dist_entropy', 0))

            if data["save_every_x_episode"] > 0 and (episode + 1) % data["save_every_x_episode"] == 0:
                checkpoint_started_at = time.perf_counter()
                checkpoint_path = _save_checkpoint(
                    policy, data, episode + 1, run_directory
                )
                total_checkpoint_seconds += time.perf_counter() - checkpoint_started_at
                logging.info("Saved checkpoint: %s", checkpoint_path)

        # Always save the final state, even when save_every is disabled or the
        # final episode is not an exact multiple of it.
        checkpoint_started_at = time.perf_counter()
        final_checkpoint = _save_checkpoint(
            policy, data, start_episode + data["episodes"], run_directory
        )
        total_checkpoint_seconds += time.perf_counter() - checkpoint_started_at
        logging.info("Final model saved to: %s", final_checkpoint)
        logging.info("Latest model saved to: %s", run_directory / 'r_mappo_latest.pt')

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

        logging.info("Overall timing summary")
        logging.info("Total runtime: %s (100.00%%)", _format_duration(total_runtime))
        logging.info("Engine start/reset: %s , %6.2f", _format_duration(total_engine_start_seconds), percentage(total_engine_start_seconds))
        logging.info("Engine steps/waiting: %s , %6.2f", _format_duration(total_engine_step_seconds), percentage(total_engine_step_seconds))
        logging.info("Policy inference: %s , %6.2f", _format_duration(total_inference_seconds), percentage(total_inference_seconds))
        logging.info("PPO updates: %s , %6.2f", _format_duration(total_training_seconds), percentage(total_training_seconds))
        logging.info("Checkpoint writing: %s , %6.2f", _format_duration(total_checkpoint_seconds), percentage(total_checkpoint_seconds))
        logging.info("Other Python/logging: %s , %6.2f",  _format_duration(other_seconds), percentage(other_seconds))
        
    except Exception as e:
            logging.error("Exception occurred during simulated 3v3 training!")
            logging.error("TYPE: %s",type(e).__name__)
            logging.error("MSG : %s",e)
            logging.warning("TRACEBACK:")
            logging.warning(traceback.print_exc)
            raise
    
    finally:
        if env is not None:
            try:
                env.close()
                logging.debug("env.close() done")
                success = True
            except Exception as close_err:
                logging.warning("env.close() failed: %s", close_err)

        if success:
            print("Simulated 3v3 pawn training complete!", flush=True)
        else:
            print("Simulated 3v3 pawn training aborted due to an error.", flush=True)
   
if __name__ == "__main__":
    main()