import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.environment.environmentpool import EnvironmentPool
from src.train.mappo.obs_r_mappo import (
    OBS_DIM,
    _prepare_reward,
)
from src.train.mappo.policy_runner import MAPPOPolicyRunner
from src.train.mappo.r_mappo import R_MAPPO
from src.train.mappo.trainerargs import TrainerArgs
from src.train.replay_buffer import SharedReplayBuffer

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


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

def configure_logging(
    config: dict[str, Any],
    timestamp: str,
) -> None:
    logging_config = config.get("logging", {})

    level_name = str(
        logging_config.get("level", "INFO")
    ).upper()

    level = getattr(logging, level_name, None)

    if not isinstance(level, int):
        raise ValueError(
            f"Invalid logging level: {level_name!r}"
        )

    log_directory = (
        Path("src/train/experiments")
        / config["run_name"]
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = log_directory / (
        f"{config['run_name']}_{timestamp}.log"
    )

    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(filename)s:%(lineno)d - "
            "%(funcName)s - %(message)s"
        ),
        level=level,
        force=True,
    )

    module_levels = logging_config.get("modules", {})

    if not isinstance(module_levels, dict):
        raise ValueError(
            "'logging.modules' must be a JSON object."
        )

    for module_name, module_level_name in module_levels.items():
        normalized_level_name = str(
            module_level_name
        ).upper()

        module_level = getattr(
            logging,
            normalized_level_name,
            None,
        )

        if not isinstance(module_level, int):
            raise ValueError(
                f"Invalid logging level "
                f"{module_level_name!r} for module "
                f"{module_name!r}."
            )

        logging.getLogger(module_name).setLevel(
            module_level
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


def load_training_config(path: Path):
    with open(
        path,
        "r",
    ) as configfile:
        return json.load(configfile)

def main() -> None:
    config_path = Path(
        "src/train/experiments/3v3Test/3v3Test.json"
    )
    data = load_training_config(config_path)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    configure_logging(
        config=data,
        timestamp=timestamp,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting training")

    if data["device"] == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(data["device"])

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was configured, but CUDA is unavailable."
        )

    device_description = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )

    logger.info(
        "Using device: %s",
        device_description,
    )

    run_directory = (
        Path("src/train/experiments")
        / data["run_name"]
        / "run"
    )

    logger.info(
        "run_name=%s",
        data["run_name"],
    )
    logger.info(
        "run_directory=%s",
        run_directory,
    )

    run_started_at = time.perf_counter()

    total_engine_start_seconds = 0.0
    total_engine_step_seconds = 0.0
    total_inference_seconds = 0.0
    total_training_seconds = 0.0
    total_checkpoint_seconds = 0.0

    success = False
    pool = EnvironmentPool(config_path)

    try:
        logger.info("Creating EnvironmentPool")

        pool_started_at = time.perf_counter()
        pool.initialize_environements()
        total_engine_start_seconds += (
            time.perf_counter() - pool_started_at
        )

        logger.info(
            "EnvironmentPool created with %d environment(s)",
            len(pool.environments),
        )

        if not pool.environments:
            raise RuntimeError(
                "EnvironmentPool contains no environments."
            )

        num_agents = int(data["number_of_agents"])
        action_dimension = int(
            data["action_dimension"]
        )
        buffer_size = int(data["buffer_size"])
        maximum_steps = int(data["maximum_steps"])

        runner = MAPPOPolicyRunner.from_config(
            data,
            device,
        )

        policy = runner.policy

        logger.info(
            "Created R-MAPPO policy runner: "
            "agents=%d action_dimension=%d team_id=%d",
            num_agents,
            action_dimension,
            int(data["training_team_id"]),
        )

        continue_training = data.get(
            "continue_training"
        )

        if not isinstance(continue_training, bool):
            raise ValueError(
                "'continue_training' must be a boolean."
            )

        start_episode = 0

        if continue_training:
            checkpoint_path = (
                run_directory
                / "r_mappo_latest.pt"
            )

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "Cannot continue training: "
                    f"checkpoint not found at {checkpoint_path}"
                )

            start_episode = runner.load_checkpoint(
                checkpoint_path,
                load_optimizers=True,
            )

            logger.info(
                "Resumed training from %s at episode %d",
                checkpoint_path,
                start_episode,
            )

        buffer = SharedReplayBuffer(
            num_agents=num_agents,
            obs_shape=(OBS_DIM,),
            action_shape=(2,),
            buffer_size=buffer_size,
            device=device,
            action_dim=action_dimension,
        )

        logger.info(
            "Created SharedReplayBuffer: capacity=%d",
            buffer_size,
        )

        trainer_args = TrainerArgs()
        trainer_args.num_mini_batch = int(
            data["mini_batch"]
        )

        trainer = R_MAPPO(
            trainer_args,
            policy,
            device=device,
        )

        recurrent_n = 1
        hidden_size = 1

        final_episode = (
            start_episode
            + int(data["episodes"])
        )

        for episode in range(
            start_episode,
            final_episode,
        ):
            episode_number = episode + 1

            logger.info(
                "Starting episode %d of %d",
                episode_number,
                final_episode,
            )

            reset_started_at = time.perf_counter()

            reset_result = (
                pool.environments[0].reset()
            )

            total_engine_start_seconds += (
                time.perf_counter()
                - reset_started_at
            )

            (
                obs_raw,
                share_obs_raw,
                available_actions_raw,
            ) = _parse_reset_result(
                reset_result
            )

            (
                obs,
                share_obs,
                available_actions,
            ) = runner.prepare_observations(
                obs_raw,
                share_obs_raw,
                available_actions_raw,
            )

            episode_reward = 0.0
            done = False
            step_count = 0

            action_counts = np.zeros(
                action_dimension,
                dtype=np.int64,
            )
            action_trace: list[list[int]] = []

            episode_won = False
            episode_lost = False
            final_ally_alive = None
            final_enemy_alive = None
            enemy_damage_done = 0.0
            own_damage_taken = 0.0
            last_info_keys: list[str] = []
            final_step_info: Any = None

            (
                rnn_states,
                rnn_states_critic,
            ) = _make_rnn_state_arrays(
                num_agents,
                recurrent_n=recurrent_n,
                hidden_size=hidden_size,
            )

            trainer.prep_rollout()

            while not done:
                if step_count >= maximum_steps + 1:
                    raise RuntimeError(
                        "Environment exceeded maximum_steps "
                        "without returning terminated=True or "
                        "truncated=True. "
                        f"episode={episode_number}, "
                        f"step_count={step_count}, "
                        f"maximum_steps={maximum_steps}."
                    )

                inference_started_at = (
                    time.perf_counter()
                )

                (
                    value_preds,
                    actions,
                    action_log_probs,
                    candidate_target_features,
                    candidate_target_mask,
                ) = runner.sample_actions(
                    obs_raw=obs_raw,
                    obs=obs,
                    share_obs=share_obs,
                    available_actions=available_actions,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                inference_elapsed = (
                    time.perf_counter()
                    - inference_started_at
                )
                total_inference_seconds += (
                    inference_elapsed
                )

                action_types = [
                    int(action)
                    for action in actions[:, 0]
                ]

                action_trace.append(action_types)

                for action_type in action_types:
                    if (
                        0
                        <= action_type
                        < action_dimension
                    ):
                        action_counts[
                            action_type
                        ] += 1
                    else:
                        logger.warning(
                            "Sampled invalid action %d; "
                            "expected range [0, %d]",
                            action_type,
                            action_dimension - 1,
                        )

                environment_started_at = (
                    time.perf_counter()
                )

                step_result = (
                    runner.step_environment(
                        pool.environments[0],
                        actions,
                        obs_raw,
                    )
                )

                engine_step_elapsed = (
                    time.perf_counter()
                    - environment_started_at
                )

                total_engine_step_seconds += (
                    engine_step_elapsed
                )

                (
                    next_obs_raw,
                    next_share_obs_raw,
                    reward,
                    terminated,
                    truncated,
                    step_info,
                    next_available_actions_raw,
                ) = _parse_step_result(
                    step_result
                )

                if (
                    not isinstance(terminated, bool)
                    or not isinstance(truncated, bool)
                ):
                    raise TypeError(
                        "Environment termination flags must "
                        "be bool values."
                    )

                if terminated and truncated:
                    raise RuntimeError(
                        "Environment returned terminated=True "
                        "and truncated=True simultaneously."
                    )

                final_step_info = step_info

                debug_metrics = (
                    _extract_episode_debug_from_info(
                        step_info
                    )
                )

                episode_won = (
                    episode_won
                    or debug_metrics["won"]
                )
                episode_lost = (
                    episode_lost
                    or debug_metrics["lost"]
                )

                if (
                    debug_metrics["ally_alive"]
                    is not None
                ):
                    final_ally_alive = (
                        debug_metrics["ally_alive"]
                    )

                if (
                    debug_metrics["enemy_alive"]
                    is not None
                ):
                    final_enemy_alive = (
                        debug_metrics["enemy_alive"]
                    )

                enemy_damage_done += (
                    debug_metrics[
                        "enemy_damage_done"
                    ]
                )
                own_damage_taken += (
                    debug_metrics[
                        "own_damage_taken"
                    ]
                )

                if debug_metrics["info_keys"]:
                    last_info_keys = (
                        debug_metrics["info_keys"]
                    )

                done = bool(
                    terminated or truncated
                )

                dones = np.full(
                    num_agents,
                    done,
                    dtype=bool,
                )

                if done and next_obs_raw is None:
                    # Falls das Environment bei Terminalzuständen
                    # keine Observation liefert, bleibt der letzte
                    # valide Zustand für die Buffer-Form erhalten.
                    next_obs = obs.copy()
                    next_share_obs = share_obs.copy()
                    next_available_actions = (
                        available_actions.copy()
                    )
                else:
                    (
                        next_obs,
                        next_share_obs,
                        next_available_actions,
                    ) = runner.prepare_observations(
                        next_obs_raw,
                        next_share_obs_raw,
                        next_available_actions_raw,
                    )

                reward_arr = _prepare_reward(
                    reward,
                    num_agents,
                )

                masks = np.ones(
                    (num_agents, 1),
                    dtype=np.float32,
                )

                # Nur natürliche Terminalzustände stoppen
                # das Value-Bootstrapping.
                if terminated:
                    masks[:] = 0.0

                if done:
                    rnn_states[:] = 0.0
                    rnn_states_critic[:] = 0.0

                active_masks = np.ones(
                    (num_agents, 1),
                    dtype=np.float32,
                )
                active_masks[dones] = 0.0

                if done:
                    active_masks[:] = 1.0

                buffer.insert(
                    share_obs=share_obs,
                    obs=obs,
                    rnn_states=rnn_states,
                    rnn_states_critic=(
                        rnn_states_critic
                    ),
                    actions=actions,
                    action_log_probs=(
                        action_log_probs
                    ),
                    value_preds=value_preds,
                    rewards=reward_arr,
                    masks=masks,
                    active_masks=active_masks,
                    available_actions=(
                        available_actions
                    ),
                    candidate_target_features=(
                        candidate_target_features
                    ),
                    candidate_target_mask=(
                        candidate_target_mask
                    ),
                )

                episode_reward += float(
                    reward_arr.mean()
                )
                step_count += 1

                obs_raw = next_obs_raw
                obs = next_obs
                share_obs = next_share_obs
                available_actions = (
                    next_available_actions
                )

            if not done:
                raise RuntimeError(
                    f"Episode {episode_number} left the "
                    "rollout loop without termination."
                )

            if step_count <= 0:
                raise RuntimeError(
                    f"Episode {episode_number} ended "
                    "without executing a step."
                )

            if final_enemy_alive is not None:
                try:
                    if int(final_enemy_alive) <= 0:
                        episode_won = True
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid final_enemy_alive value: %r",
                        final_enemy_alive,
                    )

            if final_ally_alive is not None:
                try:
                    if int(final_ally_alive) <= 0:
                        episode_lost = True
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid final_ally_alive value: %r",
                        final_ally_alive,
                    )

            # Bei natürlichem Terminalzustand ist kein Bootstrap
            # erlaubt. Bei Truncation wird der letzte Critic-Wert
            # verwendet.
            if terminated:
                next_value = np.zeros(
                    (num_agents, 1),
                    dtype=np.float32,
                )
            else:
                next_value = runner.predict_values(
                    share_obs
                )

            try:
                buffer.compute_returns(
                    next_value,
                    gamma=float(data["gamma"]),
                )
            except TypeError:
                buffer.compute_returns(
                    next_value
                )

            trainer.prep_training()

            training_started_at = (
                time.perf_counter()
            )

            train_info = trainer.train(
                buffer,
                update_actor=True,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            total_training_seconds += (
                time.perf_counter()
                - training_started_at
            )

            buffer.reset()

            average_reward_per_step = (
                episode_reward / step_count
            )

            logger.info(
                "Episode %d of %d completed",
                episode_number,
                final_episode,
            )
            logger.info(
                "Average Reward per Step: %.4f",
                average_reward_per_step,
            )
            logger.info(
                "Episode Reward: %.4f",
                episode_reward,
            )
            logger.info(
                "Steps: %d",
                step_count,
            )
            logger.info(
                "Terminated: %s",
                terminated,
            )
            logger.info(
                "Truncated: %s",
                truncated,
            )
            logger.info(
                "Won: %s",
                episode_won,
            )
            logger.info(
                "Lost: %s",
                episode_lost,
            )
            logger.info(
                "Final Ally Alive: %s",
                final_ally_alive,
            )
            logger.info(
                "Final Enemy Alive: %s",
                final_enemy_alive,
            )
            logger.info(
                "Enemy Damage Done: %.4f",
                enemy_damage_done,
            )
            logger.info(
                "Own Damage Taken: %.4f",
                own_damage_taken,
            )
            logger.info(
                "Action Counts: %s",
                action_counts.tolist(),
            )
            logger.debug(
                "Action Trace: %s",
                action_trace,
            )
            logger.info(
                "Last Info Keys: %s",
                last_info_keys,
            )
            logger.info(
                "Final Step Info: %s",
                final_step_info,
            )
            logger.info(
                "Value Loss: %.6f",
                train_info.get(
                    "value_loss",
                    0.0,
                ),
            )
            logger.info(
                "Policy Loss: %.6f",
                train_info.get(
                    "policy_loss",
                    0.0,
                ),
            )
            logger.info(
                "Entropy: %.6f",
                train_info.get(
                    "dist_entropy",
                    0.0,
                ),
            )

            save_interval = int(
                data["save_every_x_episode"]
            )

            if (
                save_interval > 0
                and episode_number
                % save_interval == 0
            ):
                checkpoint_started_at = (
                    time.perf_counter()
                )

                checkpoint_path = _save_checkpoint(
                    policy,
                    data,
                    episode_number,
                    run_directory,
                )

                total_checkpoint_seconds += (
                    time.perf_counter()
                    - checkpoint_started_at
                )

                logger.info(
                    "Saved checkpoint: %s",
                    checkpoint_path,
                )

        checkpoint_started_at = (
            time.perf_counter()
        )

        final_checkpoint = _save_checkpoint(
            policy,
            data,
            final_episode,
            run_directory,
        )

        total_checkpoint_seconds += (
            time.perf_counter()
            - checkpoint_started_at
        )

        logger.info(
            "Final model saved to: %s",
            final_checkpoint,
        )
        logger.info(
            "Latest model saved to: %s",
            run_directory / "r_mappo_latest.pt",
        )

        total_runtime = (
            time.perf_counter()
            - run_started_at
        )

        measured_runtime = (
            total_engine_start_seconds
            + total_engine_step_seconds
            + total_inference_seconds
            + total_training_seconds
            + total_checkpoint_seconds
        )

        other_seconds = max(
            0.0,
            total_runtime - measured_runtime,
        )

        def percentage(seconds: float) -> float:
            if total_runtime <= 0.0:
                return 0.0

            return (
                100.0
                * seconds
                / total_runtime
            )

        logger.info("Overall timing summary")
        logger.info(
            "Total runtime: %s (100.00%%)",
            _format_duration(total_runtime),
        )
        logger.info(
            "Engine start/reset: %s, %.2f%%",
            _format_duration(
                total_engine_start_seconds
            ),
            percentage(
                total_engine_start_seconds
            ),
        )
        logger.info(
            "Engine steps/waiting: %s, %.2f%%",
            _format_duration(
                total_engine_step_seconds
            ),
            percentage(
                total_engine_step_seconds
            ),
        )
        logger.info(
            "Policy inference: %s, %.2f%%",
            _format_duration(
                total_inference_seconds
            ),
            percentage(
                total_inference_seconds
            ),
        )
        logger.info(
            "PPO updates: %s, %.2f%%",
            _format_duration(
                total_training_seconds
            ),
            percentage(
                total_training_seconds
            ),
        )
        logger.info(
            "Checkpoint writing: %s, %.2f%%",
            _format_duration(
                total_checkpoint_seconds
            ),
            percentage(
                total_checkpoint_seconds
            ),
        )
        logger.info(
            "Other Python/logging: %s, %.2f%%",
            _format_duration(other_seconds),
            percentage(other_seconds),
        )

        success = True

    except Exception:
        logger.exception(
            "Exception occurred during simulated "
            "3v3 training."
        )
        raise

    finally:
        pool.close_environements()

        if success:
            print(
                "Simulated 3v3 pawn training complete!",
                flush=True,
            )
        else:
            print(
                "Simulated 3v3 pawn training aborted "
                "due to an error.",
                flush=True,
            )

if __name__ == "__main__":
    main()