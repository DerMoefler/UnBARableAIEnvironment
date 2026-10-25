import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import bar_ai

from src.environment.bar_environment import BAR_Environment
from src.train.mappo.policy import R_MAPPO_Policy
from src.train.replay_buffer import SharedReplayBuffer
from src.train.mappo.r_mappo import R_MAPPO
from src.train.mappo.obs_r_mappo import (
    OBS_DIM,
    UNIT_SLOT_FEATURE_DIM,
    build_r_mappo_action_context,
    _prepare_obs,
    _prepare_share_obs,
)
from src.train.mappo.trainerargs import TrainerArgs
from src.train.reward import _prepare_reward


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _make_bar_action(
    action_index: Any,
    unit_id: Any,
    target_unit_id: Any = 4211,
) -> bar_ai.Action:
    """Convert a policy action into a BAR action with an explicit target."""
    action_value = int(action_index) + 1
    unit_value = int(unit_id)
    target_value = int(target_unit_id)

    action_id = (
        bar_ai.ActionId(action_value)
        if hasattr(bar_ai, "ActionId")
        else action_value
    )

    # Prefer named construction to avoid swapping integer arguments.
    for kwargs in (
        {
            "action_id": action_id,
            "unit_id": unit_value,
            "target_unit_id": target_value,
        },
        {
            "actionId": action_id,
            "unitId": unit_value,
            "targetUnitId": target_value,
        },
    ):
        try:
            return bar_ai.Action(**kwargs)
        except TypeError:
            pass

    # Try positional construction with the target as the third argument.
    try:
        return bar_ai.Action(action_id, unit_value, target_value)
    except TypeError:
        pass

    # Fall back to the original two-argument constructor.
    try:
        result = bar_ai.Action(action_id, unit_value)
    except TypeError:
        try:
            result = bar_ai.Action()
        except TypeError as exc:
            raise TypeError(
                "Could not construct bar_ai.Action with action, unit, "
                "and target fields."
            ) from exc

    # Support both common naming conventions for bound fields.
    field_options = (
        (("action_id", "actionId"), action_id),
        (("unit_id", "unitId"), unit_value),
        (("target_unit_id", "targetUnitId"), target_value),
    )

    for names, value in field_options:
        field = next(
            (name for name in names if hasattr(result, name)),
            None,
        )
        if field is None:
            raise TypeError(
                f"bar_ai.Action does not expose any of these fields: {names}."
            )
        try:
            setattr(result, field, value)
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                f"Could not set bar_ai.Action field {field!r}."
            ) from exc

    return result


def _step_all_agents(
    env: Any,
    actions: np.ndarray,
    obs_raw: Any,
    training_team_id: int,
    policy: Any,
) -> Any:
    """Resolve policy target indices and submit BAR actions."""
    action_rows = np.asarray(actions)

    if action_rows.ndim == 1:
        action_rows = action_rows.reshape(1, -1)

    if action_rows.ndim != 2 or action_rows.shape[1] != 2:
        raise ValueError(
            "Policy actions must have shape (num_agents, 2), got "
            f"{action_rows.shape}."
        )

    own_ids, candidate_ids, _, candidate_mask = (
        build_r_mappo_action_context(
            obs_raw,
            training_team_id=training_team_id,
        )
    )

    if len(own_ids) != len(action_rows):
        raise ValueError(
            f"Got {len(action_rows)} actions but {len(own_ids)} controlled units."
        )

    action_list = []

    for row, (action_type, target_index) in enumerate(action_rows):
        action_type = int(action_type)
        target_index = int(target_index)

        if action_type == policy.attack_action_index:
            if not 0 <= target_index < candidate_ids.shape[1]:
                raise ValueError(
                    f"Invalid target index {target_index} for unit {own_ids[row]}."
                )
            if not candidate_mask[row, target_index]:
                raise ValueError(
                    f"Target slot {target_index} is empty for unit {own_ids[row]}."
                )
            target_id = int(candidate_ids[row, target_index])
        else:
            target_id = 4211

        action_list.append(
            _make_bar_action(
                action_type,
                own_ids[row],
                target_id,
            )
        )

    if not action_list:
        raise ValueError("At least one agent action is required.")

    result = env.step(action_list)

    if not isinstance(result, tuple) or len(result) != 5:
        raise RuntimeError(
            "BAR_Environment.step must return "
            "(obs, reward, terminated, truncated, info)."
        )

    return result


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
    raw_obs: Any = None,
    training_team_id: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Sample actions, log probabilities and critic values for all agents.
    Acts as a bridge between the BAR environment and the R-MAPPO policy."""
    target_features_array = None
    target_mask_array = None
    candidate_target_ids = None

    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=device,
    )
    available_actions_tensor = torch.as_tensor(
        available_actions,
        dtype=torch.bool,
        device=device,
    )

    if raw_obs is not None:
        (
            own_unit_ids,
            candidate_target_ids,
            target_features,
            target_mask,
        ) = build_r_mappo_action_context(
            raw_obs,
            training_team_id=training_team_id,
        )

        if len(own_unit_ids) != num_agents:
            raise ValueError(
                "Mismatch between the number of controlled units in "
                f"raw_obs ({len(own_unit_ids)}) and num_agents "
                f"({num_agents})."
            )

        target_features_array = np.asarray(
            target_features,
            dtype=np.float32,
        )
        target_mask_array = np.asarray(
            target_mask,
            dtype=np.bool_,
        )

        target_features_tensor = torch.as_tensor(
            target_features_array,
            dtype=torch.float32,
            device=device,
        )
        target_mask_tensor = torch.as_tensor(
            target_mask_array,
            dtype=torch.bool,
            device=device,
        )

        # Sample once, preserving target selection and its log probability.
        actions, action_log_probs = policy.get_action(
            obs_tensor,
            target_features=target_features_tensor,
            candidate_target_ids=candidate_target_ids,
            target_mask=target_mask_tensor,
            available_actions=available_actions_tensor,
        )
    else:
        actions, action_log_probs = policy.get_action(
            obs_tensor,
            available_actions=available_actions_tensor,
        )

    action_np = (
        actions.detach().cpu().numpy()
        if isinstance(actions, torch.Tensor)
        else np.asarray(actions)
    )
    action_log_prob_np = (
        action_log_probs.detach().cpu().numpy()
        if isinstance(action_log_probs, torch.Tensor)
        else np.asarray(action_log_probs)
    )

    action_np = action_np.reshape(num_agents, 2).astype(np.float32)
    action_log_prob_np = action_log_prob_np.reshape(
        num_agents, 1
    ).astype(np.float32)

    with torch.no_grad():
        share_obs_tensor = torch.as_tensor(
            share_obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        values = policy.critic(share_obs_tensor)

    return (
        _repeat_value(values, num_agents),
        action_np,
        action_log_prob_np,
        target_features_array,
        target_mask_array,
    )




def _parse_reset_result(
    reset_result: Any,
) -> tuple[Any, Any, Any]:
    """
    Normalize environment reset results.

    Supported formats:
    - (obs, info)
    - (obs, share_obs, available_actions)
    - (obs, share_obs, available_actions, info)
    - obs
    """
    if not isinstance(reset_result, tuple):
        return reset_result, None, None

    if len(reset_result) == 0:
        raise RuntimeError("env.reset() returned an empty tuple.")

    obs_raw = reset_result[0]
    share_obs_raw = None
    available_actions_raw = None

    if len(reset_result) == 2:
        # Standard Gymnasium: (obs, info)
        # A dictionary in position 2 is metadata, not share_obs.
        second = reset_result[1]

        if not isinstance(second, dict):
            share_obs_raw = second

    elif len(reset_result) == 3:
        # Custom format: (obs, share_obs, available_actions)
        share_obs_raw = reset_result[1]
        available_actions_raw = reset_result[2]

    elif len(reset_result) >= 4:
        # Custom format: (obs, share_obs, available_actions, info)
        share_obs_raw = reset_result[1]
        available_actions_raw = reset_result[2]

    return obs_raw, share_obs_raw, available_actions_raw



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

def load_training_config():
    with open(
        "src/train/experiments/3v3Test/3v3Test.json",
        "r",
    ) as configfile:
        return json.load(configfile)

def main() -> None:
    #import Json configfile
    data = load_training_config()

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
            target_feature_dim=UNIT_SLOT_FEATURE_DIM,
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
            action_dim=action_dim,
        )

        logging.info("created shared replay buffer")

        trainer_args = TrainerArgs()
        trainer_args.num_mini_batch = data["mini_batch"]

        trainer = R_MAPPO(
            trainer_args,
            policy,
            device=device,
        )


        recurrent_n = 1
        hidden_size = 1
    


        for episode in range(start_episode, start_episode + data["episodes"]):
            logging.debug("starting episode %d", episode + 1)
            engine_step_seconds = 0.0

            logging.info("starting environment reset")
            reset_result = env.reset()
            logging.info("reset complete")


            obs_raw, share_obs_raw, available_actions_raw = _parse_reset_result(
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

                value_preds, actions, action_log_probs, candidate_target_features, candidate_target_mask = (
                    _policy_sample_actions(
                        policy=policy,
                        obs=obs,
                        share_obs=share_obs,
                        available_actions=available_actions,
                        device=device,
                        num_agents=data["number_of_agents"],
                        raw_obs=obs_raw,
                        training_team_id=data["training_team_id"],
                    )
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
                step_result = _step_all_agents(
                    env,
                    env_actions,
                    obs_raw=obs_raw,
                        training_team_id=data["training_team_id"],
                    policy=policy,
                )
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
                    active_masks=active_masks,
                    available_actions=available_actions,
                    candidate_target_features=candidate_target_features,
                    candidate_target_mask=candidate_target_mask,
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

            logging.info("Episode %d of %d", episode, data["episodes"])
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

        success = True
        
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
            except Exception as close_err:
                logging.warning("env.close() failed: %s", close_err)

        if success:
            print("Simulated 3v3 pawn training complete!", flush=True)
        else:
            print("Simulated 3v3 pawn training aborted due to an error.", flush=True)
   
if __name__ == "__main__":
    main()