from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.environment.environmentpool import EnvironmentPool
from src.train.mappo.obs_r_mappo import (
    PreparedStep,
    RMAPPOObservationAdapter,
)
from src.train.mappo.policy_runner import (
    MAPPOPolicyRunner,
)
from src.train.mappo.r_mappo import R_MAPPO
from src.train.mappo.trainerargs import TrainerArgs
from src.train.rollout_buffer import RolloutBuffer


logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _make_rnn_state_arrays(
    num_agents: int,
    recurrent_n: int = 1,
    hidden_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create placeholder actor and critic RNN states.

    The current policy is feed-forward. These zero arrays only satisfy
    the replay-buffer and MAPPO interfaces.
    """
    rnn_shape = (
        num_agents,
        recurrent_n,
        hidden_size,
    )

    return (
        np.zeros(
            rnn_shape,
            dtype=np.float32,
        ),
        np.zeros(
            rnn_shape,
            dtype=np.float32,
        ),
    )


def _episode_metrics(
    info: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract episode metrics from BAR_Environment info.
    """
    outcome = info.get("episode_outcome")

    return {
        "won": outcome == "won",
        "lost": outcome == "lost",
        "own_alive_count": info.get(
            "own_alive_count"
        ),
        "enemy_alive_count": info.get(
            "enemy_alive_count"
        ),
        "enemy_damage_done": float(
            info.get(
                "reward_info",
                {},
            ).get(
                "enemy_damage_done",
                0.0,
            )
            if isinstance(
                info.get("reward_info"),
                dict,
            )
            else 0.0
        ),
        "own_damage_taken": float(
            info.get(
                "reward_info",
                {},
            ).get(
                "own_damage_taken",
                0.0,
            )
            if isinstance(
                info.get("reward_info"),
                dict,
            )
            else 0.0
        ),
        "info_keys": sorted(info.keys()),
    }


# ---------------------------------------------------------------------------
# Configuration and logging
# ---------------------------------------------------------------------------


def load_training_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "Training configuration must be a JSON object."
        )

    training_config = config.get("training")

    if not isinstance(training_config, dict):
        raise ValueError(
            "Training configuration must contain a "
            "'training' JSON object."
        )

    mappo_config = training_config.get("mappo")

    if not isinstance(mappo_config, dict):
        raise ValueError(
            "'training.mappo' must be a JSON object."
        )

    return config, training_config, mappo_config


def configure_logging(
    config: dict[str, Any],
    timestamp: str,
) -> None:
    logging_config = config.get(
        "logging",
        {},
    )

    if not isinstance(logging_config, dict):
        raise ValueError(
            "'logging' must be a JSON object."
        )

    level_name = str(
        logging_config.get(
            "level",
            "INFO",
        )
    ).upper()

    level = getattr(
        logging,
        level_name,
        None,
    )

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
        encoding="utf-8",
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(filename)s:%(lineno)d - "
            "%(funcName)s - %(message)s"
        ),
        level=level,
        force=True,
    )

    module_levels = logging_config.get(
        "modules",
        {},
    )

    if not isinstance(module_levels, dict):
        raise ValueError(
            "'logging.modules' must be "
            "a JSON object."
        )

    for (
        module_name,
        module_level_name,
    ) in module_levels.items():
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
                "Invalid logging level "
                f"{module_level_name!r} for module "
                f"{module_name!r}."
            )

        logging.getLogger(
            module_name
        ).setLevel(
            module_level
        )


def _select_device(
    config: dict[str, Any],
) -> torch.device:
    device_config = config["device"]

    if device_config == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            device_config
        )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was configured, but CUDA "
            "is unavailable."
        )

    return device


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _save_checkpoint(
    policy: Any,
    adapter: RMAPPOObservationAdapter,
    episode: int,
    model_dir: Path,
) -> Path:
    """
    Save policy, target scorer, critic, optimizers and reconstruction
    metadata atomically.
    """
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "format_version": 2,
        "episode": int(episode),
        "obs_dim": int(adapter.obs_dim),
        "action_dim": int(
            adapter.action_dim
        ),
        "num_agents": int(
            adapter.num_agents
        ),
        "training_team_id": int(
            adapter.training_team_id
        ),
        "max_enemy_slots": int(
            adapter.max_enemy_slots
        ),
        "target_feature_dim": int(
            adapter.target_feature_dim
        ),
        "position_scale": float(
            adapter.position_scale
        ),
        "actor_state_dict": (
            policy.actor.state_dict()
        ),
        "target_scorer_state_dict": (
            policy.target_scorer.state_dict()
        ),
        "critic_state_dict": (
            policy.critic.state_dict()
        ),
        "actor_optimizer_state_dict": (
            policy.actor_optimizer.state_dict()
            if hasattr(
                policy,
                "actor_optimizer",
            )
            else None
        ),
        "critic_optimizer_state_dict": (
            policy.critic_optimizer.state_dict()
            if hasattr(
                policy,
                "critic_optimizer",
            )
            else None
        ),
    }

    episode_path = model_dir / (
        f"r_mappo_episode_{episode:04d}.pt"
    )

    temporary_path = (
        episode_path.with_suffix(".tmp")
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(
        episode_path
    )

    latest_path = (
        model_dir / "r_mappo_latest.pt"
    )

    latest_temporary_path = (
        latest_path.with_suffix(".tmp")
    )

    torch.save(
        checkpoint,
        latest_temporary_path,
    )

    latest_temporary_path.replace(
        latest_path
    )

    return episode_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config_path = Path(
        "src/train/experiments/3v3Test/"
        "3v3Test.json"
    )

    config, training_config, mappo_config = load_training_config(
        config_path
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    configure_logging(
        config=config,
        timestamp=timestamp,
    )

    logger.info("Starting training")

    device = _select_device(config)

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
        / config["run_name"]
        / "run"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "run_name=%s",
        config["run_name"],
    )
    logger.info(
        "run_directory=%s",
        run_directory,
    )

    success = False

    pool = EnvironmentPool(
        config_path
    )

    try:
        # ---------------------------------------------------------------
        # Observation adapter and policy
        # ---------------------------------------------------------------

        adapter = (
            RMAPPOObservationAdapter.from_config(
                config
            )
        )

        runner = MAPPOPolicyRunner.from_config(
            config,
            device,
            obs_dim=adapter.obs_dim,
            target_feature_dim=(
                adapter.target_feature_dim
            ),
        )

        policy = runner.policy

        logger.info(
            "Created observation adapter: "
            "obs_dim=%d target_feature_dim=%d "
            "max_enemy_slots=%d",
            adapter.obs_dim,
            adapter.target_feature_dim,
            adapter.max_enemy_slots,
        )

        logger.info(
            "Created policy runner: "
            "agents=%d actions=%d team_id=%d",
            adapter.num_agents,
            adapter.action_dim,
            adapter.training_team_id,
        )

        # ---------------------------------------------------------------
        # Environment pool
        # ---------------------------------------------------------------

        logger.info(
            "Creating EnvironmentPool"
        )

        pool.initialize_environements()

        if not pool.environments:
            raise RuntimeError(
                "EnvironmentPool contains "
                "no environments."
            )

        logger.info(
            "EnvironmentPool created with "
            "%d environment(s)",
            len(pool.environments),
        )

        # Aktuell wird nur die erste Umgebung verwendet.
        environment = pool.environments[0]

        # ---------------------------------------------------------------
        # Configuration validation
        # ---------------------------------------------------------------

        maximum_steps = int(
            training_config["maximum_steps"]
        )
        buffer_size = int(
            training_config["buffer_size"]
        )

        if maximum_steps <= 0:
            raise ValueError(
                "maximum_steps must be greater "
                "than 0."
            )

        if buffer_size < maximum_steps:
            raise ValueError(
                "buffer_size must be at least as "
                "large as maximum_steps: "
                f"buffer_size={buffer_size}, "
                f"maximum_steps={maximum_steps}."
            )

        continue_training = training_config.get(
            "continue_training"
        )

        if not isinstance(
            continue_training,
            bool,
        ):
            raise ValueError(
                "'continue_training' must be "
                "a boolean."
            )

        # ---------------------------------------------------------------
        # Optional checkpoint loading
        # ---------------------------------------------------------------

        start_episode = 0

        if continue_training:
            checkpoint_path = (
                run_directory
                / "r_mappo_latest.pt"
            )

            start_episode = (
                runner.load_checkpoint(
                    checkpoint_path,
                    load_optimizers=True,
                )
            )

            logger.info(
                "Resumed training from %s "
                "at episode %d",
                checkpoint_path,
                start_episode,
            )

        # ---------------------------------------------------------------
        # Buffer and trainer
        # ---------------------------------------------------------------

        buffer = RolloutBuffer(
            num_agents=adapter.num_agents,
            obs_shape=(
                adapter.obs_dim,
            ),
            action_shape=(2,),
            buffer_size=buffer_size,
            device=device,
            action_dim=adapter.action_dim,
        )

        logger.info(
            "Created RolloutBuffer: "
            "capacity=%d",
            buffer_size,
        )

        trainer_args = TrainerArgs()
        trainer_args.num_mini_batch = int(
            training_config["mini_batch"]
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
            + int(training_config["episodes"])
        )

        # ---------------------------------------------------------------
        # Episode loop
        # ---------------------------------------------------------------

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

            # -----------------------------------------------------------
            # Reset
            # -----------------------------------------------------------

            obs_raw, reset_info = environment.reset()

            if not isinstance(obs_raw, dict):
                raise TypeError(
                    "BAR_Environment.reset() must return the observation "
                    f"as a dictionary, got {type(obs_raw).__name__}."
                )

            if not isinstance(reset_info, dict):
                raise TypeError(
                    "BAR_Environment.reset() must return info as a dictionary, "
                    f"got {type(reset_info).__name__}."
                )

            adapter.reset_episode(obs_raw)

            prepared = adapter.prepare(
                obs_raw=obs_raw,
            )

            logger.debug(
                "Episode %d stable unit slots: %s",
                episode_number,
                adapter.agent_unit_ids,
            )

            episode_reward = 0.0
            done = False
            terminated = False
            truncated = False
            step_count = 0

            action_counts = np.zeros(
                adapter.action_dim,
                dtype=np.int64,
            )
            action_trace: list[
                list[int]
            ] = []

            episode_won = False
            episode_lost = False
            final_ally_alive = None
            final_enemy_alive = None
            enemy_damage_done = 0.0
            own_damage_taken = 0.0
            last_info_keys: list[str] = []
            final_step_info: dict[
                str,
                Any,
            ] = {}

            (
                rnn_states,
                rnn_states_critic,
            ) = _make_rnn_state_arrays(
                adapter.num_agents,
                recurrent_n=recurrent_n,
                hidden_size=hidden_size,
            )

            trainer.prep_rollout()
            runner.set_evaluation_mode()

            # -----------------------------------------------------------
            # Step loop
            # -----------------------------------------------------------

            while not done:
                if step_count > maximum_steps + 1:
                    raise RuntimeError(
                        "Environment exceeded "
                        "maximum_steps without returning "
                        "terminated=True or truncated=True: "
                        f"episode={episode_number}, "
                        f"step_count={step_count}, "
                        f"maximum_steps={maximum_steps}."
                    )

                # -------------------------------------------------------
                # Policy inference
                # -------------------------------------------------------

                policy_step = runner.act(
                    prepared
                )

                if device.type == "cuda":
                    torch.cuda.synchronize(
                        device
                    )

                action_types = [
                    int(action)
                    for action
                    in policy_step.actions[:, 0]
                ]

                action_trace.append(
                    action_types
                )

                # Nur aktive Agents in der Statistik zählen.
                for row, action_type in enumerate(
                    action_types
                ):
                    if (
                        prepared.active_mask[
                            row,
                            0,
                        ]
                        <= 0.0
                    ):
                        continue

                    if (
                        0
                        <= action_type
                        < adapter.action_dim
                    ):
                        action_counts[
                            action_type
                        ] += 1
                    else:
                        logger.warning(
                            "Sampled invalid action %d; "
                            "expected range [0, %d]",
                            action_type,
                            adapter.action_dim - 1,
                        )

                # -------------------------------------------------------
                # Convert policy output to BAR actions
                # -------------------------------------------------------

                bar_actions = (
                    runner.create_bar_actions(
                        policy_step,
                        prepared,
                    )
                )

                # -------------------------------------------------------
                # Environment step
                # -------------------------------------------------------

                (
                    next_obs_raw,
                    reward,
                    terminated,
                    truncated,
                    step_info,
                ) = environment.step(bar_actions)

                if not isinstance(terminated, bool):
                    raise TypeError(
                        "BAR_Environment.step() must return terminated as bool, "
                        f"got {type(terminated).__name__}."
                    )

                if not isinstance(truncated, bool):
                    raise TypeError(
                        "BAR_Environment.step() must return truncated as bool, "
                        f"got {type(truncated).__name__}."
                    )

                if terminated and truncated:
                    raise RuntimeError(
                        "BAR_Environment.step() returned terminated=True and "
                        "truncated=True simultaneously."
                    )

                if not isinstance(step_info, dict):
                    raise TypeError(
                        "BAR_Environment.step() must return info as a dictionary, "
                        f"got {type(step_info).__name__}."
                    )

                done = bool(
                    terminated or truncated
                )

                final_step_info = step_info

                # -------------------------------------------------------
                # Episode metrics
                # -------------------------------------------------------

                metrics = _episode_metrics(
                    step_info
                )

                episode_won = (
                    episode_won
                    or metrics["won"]
                )
                episode_lost = (
                    episode_lost
                    or metrics["lost"]
                )

                if (
                    metrics[
                        "own_alive_count"
                    ]
                    is not None
                ):
                    final_ally_alive = metrics[
                        "own_alive_count"
                    ]

                if (
                    metrics[
                        "enemy_alive_count"
                    ]
                    is not None
                ):
                    final_enemy_alive = metrics[
                        "enemy_alive_count"
                    ]

                enemy_damage_done += metrics[
                    "enemy_damage_done"
                ]
                own_damage_taken += metrics[
                    "own_damage_taken"
                ]
                last_info_keys = metrics[
                    "info_keys"
                ]

                # -------------------------------------------------------
                # Reward and masks
                # -------------------------------------------------------

                reward_array = (
                    adapter.prepare_reward(
                        reward
                    )
                )

                masks = np.ones(
                    (
                        adapter.num_agents,
                        1,
                    ),
                    dtype=np.float32,
                )

                # Nur ein natürliches Ende verhindert
                # Value-Bootstrapping.
                if terminated:
                    masks[:] = 0.0

                # active_mask gehört zum aktuellen PreparedStep,
                # für das die Action gewählt wurde.
                active_masks = (
                    prepared.active_mask.copy()
                )

                # -------------------------------------------------------
                # Store transition
                # -------------------------------------------------------

                buffer.insert(
                    share_obs=(
                        prepared.share_obs
                    ),
                    obs=prepared.obs,
                    rnn_states=rnn_states,
                    rnn_states_critic=(
                        rnn_states_critic
                    ),
                    actions=(
                        policy_step.actions
                    ),
                    action_log_probs=(
                        policy_step.action_log_probs
                    ),
                    value_preds=(
                        policy_step.value_predictions
                    ),
                    rewards=reward_array,
                    masks=masks,
                    active_masks=active_masks,
                    available_actions=(
                        prepared
                        .available_actions
                    ),
                    candidate_target_features=(
                        prepared.target_features
                    ),
                    candidate_target_mask=(
                        prepared.target_mask
                    ),
                )

                episode_reward += float(
                    reward_array.mean()
                )
                step_count += 1

                # -------------------------------------------------------
                # Prepare next state
                # -------------------------------------------------------

                next_prepared: (
                    PreparedStep | None
                ) = None

                if next_obs_raw is not None:
                    next_prepared = adapter.prepare(
                        obs_raw=next_obs_raw,
                    )

                if not done:
                    if next_prepared is None:
                        raise RuntimeError(
                            "Environment returned no "
                            "observation while the episode "
                            "is still running."
                        )

                    prepared = next_prepared

                elif truncated:
                    # Für korrektes Bootstrap bei Truncation
                    # wird der Zustand nach dem letzten Step
                    # benötigt.
                    if next_prepared is None:
                        raise RuntimeError(
                            "A truncated episode must "
                            "return a final observation "
                            "for value bootstrapping."
                        )

                    prepared = next_prepared

                if done:
                    rnn_states[:] = 0.0
                    rnn_states_critic[:] = 0.0

            # -----------------------------------------------------------
            # Validate episode end
            # -----------------------------------------------------------

            if step_count <= 0:
                raise RuntimeError(
                    f"Episode {episode_number} "
                    "ended without executing a step."
                )

            if (
                final_enemy_alive is not None
                and int(final_enemy_alive) <= 0
            ):
                episode_won = True

            if (
                final_ally_alive is not None
                and int(final_ally_alive) <= 0
            ):
                episode_lost = True

            # -----------------------------------------------------------
            # Bootstrap and returns
            # -----------------------------------------------------------

            if terminated:
                next_value = np.zeros(
                    (
                        adapter.num_agents,
                        1,
                    ),
                    dtype=np.float32,
                )
            else:
                # Truncated Episode:
                # Bootstrap aus finaler Observation.
                next_value = (
                    runner.predict_values(
                        prepared.share_obs
                    )
                )

            try:
                buffer.compute_returns(
                    next_value,
                    gamma=float(
                        mappo_config.get("gamma")
                    ),
                )
            except TypeError:
                # Kompatibilität mit älterer
                # RolloutBuffer-Signatur.
                buffer.compute_returns(
                    next_value
                )

            # -----------------------------------------------------------
            # PPO update
            # -----------------------------------------------------------

            trainer.prep_training()
            runner.set_training_mode()

            train_info = trainer.train(
                buffer,
                update_actor=True,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(
                    device
                )

            buffer.reset()

            # -----------------------------------------------------------
            # Episode logging
            # -----------------------------------------------------------

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
                float(
                    train_info.get(
                        "value_loss",
                        0.0,
                    )
                ),
            )
            logger.info(
                "Policy Loss: %.6f",
                float(
                    train_info.get(
                        "policy_loss",
                        0.0,
                    )
                ),
            )
            logger.info(
                "Entropy: %.6f",
                float(
                    train_info.get(
                        "dist_entropy",
                        0.0,
                    )
                ),
            )

            # -----------------------------------------------------------
            # Periodic checkpoint
            # -----------------------------------------------------------

            save_interval = int(
                training_config["save_every_x_episode"]
            )

            if (
                save_interval > 0
                and episode_number
                % save_interval == 0
            ):

                checkpoint_path = (
                    _save_checkpoint(
                        policy,
                        adapter,
                        episode_number,
                        run_directory,
                    )
                )

                logger.info(
                    "Saved checkpoint: %s",
                    checkpoint_path,
                )

        # ---------------------------------------------------------------
        # Final checkpoint
        # ---------------------------------------------------------------

        final_checkpoint = _save_checkpoint(
            policy,
            adapter,
            final_episode,
            run_directory,
        )

        logger.info(
            "Final model saved to: %s",
            final_checkpoint,
        )
        logger.info(
            "Latest model saved to: %s",
            run_directory
            / "r_mappo_latest.pt",
        )

        success = True

    except Exception:
        logger.exception(
            "Exception occurred during "
            "simulated 3v3 training."
        )
        raise

    finally:
        try:
            pool.close_environements()
        except Exception:
            logger.exception(
                "Closing EnvironmentPool failed."
            )

        if success:
            print(
                "Simulated 3v3 pawn training "
                "complete!",
                flush=True,
            )
        else:
            print(
                "Simulated 3v3 pawn training "
                "aborted due to an error.",
                flush=True,
            )

if __name__ == "__main__":
    main()