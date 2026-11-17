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


# ---------------------------------------------------------------------------
# Creating Adapters, rn just mappo specific
# ---------------------------------------------------------------------------

def create_mappo_adapters(number_of_environments: int, data):
    adapters: list[RMAPPOObservationAdapter] = []
    for i in range(number_of_environments):
        adapters.append(RMAPPOObservationAdapter.from_config(data))

        logger.info(
                    "Created observation adapter: %s"
                    "obs_dim=%d target_feature_dim=%d "
                    "max_enemy_slots=%d",
                    i,
                    adapters[i].obs_dim,
                    adapters[i].target_feature_dim,
                    adapters[i].max_enemy_slots,
                )

    return adapters


def _combine_rollout_buffers(
    buffers: list[RolloutBuffer],
    *,
    device: torch.device,
) -> RolloutBuffer:
    populated_buffers = [
        buffer
        for buffer in buffers
        if buffer.step > 0
    ]
    if not populated_buffers:
        raise RuntimeError(
            "Cannot combine empty environment rollout buffers."
        )

    template = populated_buffers[0]
    for buffer in populated_buffers[1:]:
        if (
            buffer.num_agents != template.num_agents
            or buffer.share_obs.shape[1:] != template.share_obs.shape[1:]
            or buffer.obs.shape[2:] != template.obs.shape[2:]
            or buffer.actions.shape[2:] != template.actions.shape[2:]
            or buffer.action_dim != template.action_dim
        ):
            raise ValueError(
                "Environment rollout buffers have incompatible shapes."
            )

    combined = RolloutBuffer(
        num_agents=template.num_agents,
        obs_shape=template.share_obs.shape[1:],
        action_shape=template.actions.shape[2:],
        buffer_size=sum(buffer.step for buffer in populated_buffers),
        device=device,
        action_dim=template.action_dim,
    )

    for source in populated_buffers:
        for source_step in range(source.step):
            combined.insert(
                share_obs=source.share_obs[source_step],
                obs=source.obs[source_step],
                rnn_states=source.rnn_states[source_step],
                rnn_states_critic=(
                    source.rnn_states_critic[source_step]
                ),
                actions=source.actions[source_step],
                action_log_probs=(
                    source.action_log_probs[source_step]
                ),
                value_preds=source.value_preds[source_step],
                rewards=source.rewards[source_step],
                masks=source.masks[source_step + 1],
                active_masks=source.active_masks[source_step + 1],
                available_actions=(
                    source.available_actions[source_step]
                ),
                candidate_target_features=(
                    None
                    if source.candidate_target_features is None
                    else source.candidate_target_features[source_step]
                ),
                candidate_target_mask=(
                    None
                    if source.candidate_target_mask is None
                    else source.candidate_target_mask[source_step]
                ),
            )
            combined_step = combined.step - 1
            combined.value_preds[combined_step] = (
                source.value_preds[source_step]
            )
            combined.returns[combined_step] = (
                source.returns[source_step]
            )

    return combined


# ---------------------------------------------------------------------------
# Configuration and logging
# ---------------------------------------------------------------------------


def load_training_config(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "Training configuration must be "
            "a JSON object."
        )

    return config


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

    data = load_training_config(
        config_path
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    configure_logging(
        config=data,
        timestamp=timestamp,
    )

    logger.info("Starting training")

    device = _select_device(data)

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

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "run_name=%s",
        data["run_name"],
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
        logger.info("Creating EnvironmentPool")

        pool.initialize_environements()

        if not pool.environments:
            raise RuntimeError("EnvironmentPool contains no environments.")

        logger.info(
            "EnvironmentPool created with "
            "%d environment(s)",
            len(pool.environments),
        )
        # ---------------------------------------------------------------
        # Observation adapter and policy, mappo specific
        # ---------------------------------------------------------------
        if data["algorithm"] == "MAPPO":
            adapters = create_mappo_adapters(data["number_of_environments"], data)

            for i in range(len(adapters)):
                adapters[i].assign_adapter_id(i)

            runner = MAPPOPolicyRunner.from_config(
                data,
                device,
                obs_dim=adapters[0].obs_dim,
                target_feature_dim=(
                    adapters[0].target_feature_dim
                ),
            )

            policy = runner.policy

            logger.info(
                        "Created policy runner: "
                        "agents=%d actions=%d team_id=%d",
                        data["number_of_agents"],
                        data["action_dimension"],
                        data["training_team_id"],
                    )
        else:
            raise RuntimeError("No proper algorithm configured")

        # ---------------------------------------------------------------
        # Configuration validation
        # ---------------------------------------------------------------

        maximum_steps = int(
            data["maximum_steps"]
        )
        buffer_size = int(
            data["buffer_size"]
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

        continue_training = data.get(
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

        logger.info(
            "Creating per-environment RolloutBuffers: "
            "capacity=%d each",
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

            for i in range(len(pool.environments)):
                obs_raw = pool.reset_environment(i)

                if not isinstance(obs_raw, dict):
                    raise TypeError(
                        "BAR_Environment.reset() must return the observation "
                        f"as a dictionary, got {type(obs_raw).__name__}."
                    )

                adapters[i].reset_episode(obs_raw)
                pool.environments[i].stepinfo.receive_prepared(
                    adapters[i].prepare(obs_raw=obs_raw)
                )

            trainer.prep_rollout()
            runner.set_evaluation_mode()

            environment_buffers = [
                RolloutBuffer(
                    num_agents=adapters[env_id].num_agents,
                    obs_shape=(adapters[env_id].obs_dim,),
                    action_shape=(2,),
                    buffer_size=buffer_size,
                    device=device,
                    action_dim=adapters[env_id].action_dim,
                )
                for env_id in range(len(pool.environments))
            ]
            recurrent_states = [
                _make_rnn_state_arrays(
                    adapters[env_id].num_agents,
                    recurrent_n=recurrent_n,
                    hidden_size=hidden_size,
                )
                for env_id in range(len(pool.environments))
            ]
            # -----------------------------------------------------------
            # Step loop
            # -----------------------------------------------------------

            while not pool.check_done():

                # -------------------------------------------------------
                # Policy inference
                # -------------------------------------------------------

                policy_steps: list[Any] = [None] * len(pool.environments)
                bar_actions: list[Any] = [None] * len(pool.environments)

                for env_id, env in enumerate(pool.environments):
                    if env.stepinfo.done or env.episode_step >= pool.maximum_steps:
                        continue

                    prepared = env.stepinfo.prepared
                    if prepared is None:
                        raise RuntimeError(f"Environment {env_id} has no prepared observation.")

                    policy_steps[env_id] = runner.act(prepared)

                    if device.type == "cuda":
                        torch.cuda.synchronize(
                            device
                        )

                    action_types = [
                        int(action)
                        for action
                        in policy_steps[env_id].actions[:, 0]
                    ]

                    for row, action_type in enumerate(action_types):
                        if (
                            prepared.active_mask[
                                row,
                                0,
                            ]
                            <= 0.0
                        ):
                            continue

                        if not 0 <= action_type < adapters[env_id].action_dim:
                            logger.warning(
                                "Sampled invalid action %d; "
                                "expected range [0, %d]",
                                action_type,
                                adapters[env_id].action_dim - 1,
                            )

                # -------------------------------------------------------
                # Convert policy output to BAR actions
                # -------------------------------------------------------

                    bar_actions[env_id] = (
                        runner.create_bar_actions(
                            policy_steps[env_id],
                            prepared,
                        )
                    )

                # -------------------------------------------------------
                # Environment step
                # -------------------------------------------------------

                step_results = pool.step_environments(bar_actions)
                if len(step_results) != len(pool.environments):
                    raise RuntimeError(
                        "EnvironmentPool returned a different number of "
                        "results than environments."
                    )

                for env_id, result in enumerate(step_results):
                    environment = pool.environments[env_id]
                    stepinfo = environment.stepinfo
                    if result is None:
                        if not stepinfo.done:
                            raise RuntimeError(
                                f"Environment {env_id} did not step and is "
                                "not marked done."
                            )
                        continue

                    (
                        next_obs_raw,
                        reward,
                        terminated,
                        truncated,
                        step_info,
                    ) = result

                    if not isinstance(terminated, bool):
                        raise TypeError(
                            "BAR_Environment.step() must return terminated "
                            f"as bool, got {type(terminated).__name__}."
                        )
                    if not isinstance(truncated, bool):
                        raise TypeError(
                            "BAR_Environment.step() must return truncated "
                            f"as bool, got {type(truncated).__name__}."
                        )
                    if terminated and truncated:
                        raise RuntimeError(
                            "BAR_Environment.step() returned terminated=True "
                            "and truncated=True simultaneously."
                        )
                    if not isinstance(step_info, dict):
                        raise TypeError(
                            "BAR_Environment.step() must return info as a "
                            f"dictionary, got {type(step_info).__name__}."
                        )

                    policy_step = policy_steps[env_id]
                    prepared = stepinfo.prepared
                    if policy_step is None or prepared is None:
                        raise RuntimeError(
                            f"Environment {env_id} returned a step result "
                            "without a corresponding policy action."
                        )

                    action_types = [
                        int(action)
                        for action in policy_step.actions[:, 0]
                    ]
                    stepinfo.update_stepinfo(
                        episode_number=episode_number,
                        action_types=action_types,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=step_info,
                        active_mask=prepared.active_mask,
                    )

                    next_prepared: PreparedStep | None = None
                    if next_obs_raw is not None:
                        if not isinstance(next_obs_raw, dict):
                            raise TypeError(
                                "BAR_Environment.step() must return the "
                                "observation as a dictionary, got "
                                f"{type(next_obs_raw).__name__}."
                            )
                        next_prepared = adapters[env_id].prepare(
                            obs_raw=next_obs_raw,
                        )

                    if not stepinfo.done and next_prepared is None:
                        raise RuntimeError(
                            f"Environment {env_id} returned no observation "
                            "while its episode is still running."
                        )
                    if truncated and next_prepared is None:
                        raise RuntimeError(
                            f"Truncated environment {env_id} must return its "
                            "final observation for value bootstrapping."
                        )
                    if next_prepared is not None:
                        stepinfo.receive_prepared(next_prepared)

                    reward_array = adapters[env_id].prepare_reward(reward)
                    masks = np.ones(
                        (adapters[env_id].num_agents, 1),
                        dtype=np.float32,
                    )
                    if terminated:
                        masks[:] = 0.0

                    rnn_states, rnn_states_critic = recurrent_states[env_id]
                    environment_buffers[env_id].insert(
                        share_obs=prepared.share_obs,
                        obs=prepared.obs,
                        rnn_states=rnn_states,
                        rnn_states_critic=rnn_states_critic,
                        actions=policy_step.actions,
                        action_log_probs=policy_step.action_log_probs,
                        value_preds=policy_step.value_predictions,
                        rewards=reward_array,
                        masks=masks,
                        active_masks=prepared.active_mask.copy(),
                        available_actions=prepared.available_actions,
                        candidate_target_features=prepared.target_features,
                        candidate_target_mask=prepared.target_mask,
                    )

            # Each environment's returns are computed independently so GAE
            # never follows a transition from a different environment.
            for env_id, environment in enumerate(pool.environments):
                stepinfo = environment.stepinfo
                if not stepinfo.done:
                    raise RuntimeError(
                        f"Environment {env_id} ended the batch without "
                        "terminating or truncating its episode."
                    )
                if environment_buffers[env_id].step <= 0:
                    raise RuntimeError(
                        f"Environment {env_id} ended without executing a step."
                    )

                if stepinfo.terminated:
                    next_value = np.zeros(
                        (adapters[env_id].num_agents, 1),
                        dtype=np.float32,
                    )
                else:
                    final_prepared = stepinfo.prepared
                    if final_prepared is None:
                        raise RuntimeError(
                            f"Truncated environment {env_id} has no final "
                            "observation for value bootstrapping."
                        )
                    next_value = runner.predict_values(
                        final_prepared.share_obs
                    )

                environment_buffers[env_id].compute_returns(
                    next_value,
                    gamma=float(data["gamma"]),
                )

            buffer = _combine_rollout_buffers(
                environment_buffers,
                device=device,
            )

            episode_reward = sum(
                environment.stepinfo.episode_reward
                for environment in pool.environments
            )
            step_count = sum(
                environment.stepinfo.step_count
                for environment in pool.environments
            )
            episode_won = any(
                environment.stepinfo.episode_won
                for environment in pool.environments
            )
            episode_lost = any(
                environment.stepinfo.episode_lost
                for environment in pool.environments
            )
            ally_counts = [
                environment.stepinfo.final_ally_alive
                for environment in pool.environments
                if environment.stepinfo.final_ally_alive is not None
            ]
            enemy_counts = [
                environment.stepinfo.final_enemy_alive
                for environment in pool.environments
                if environment.stepinfo.final_enemy_alive is not None
            ]
            final_ally_alive = sum(ally_counts) if ally_counts else None
            final_enemy_alive = sum(enemy_counts) if enemy_counts else None
            enemy_damage_done = sum(
                environment.stepinfo.enemy_damage_done
                for environment in pool.environments
            )
            own_damage_taken = sum(
                environment.stepinfo.own_damage_taken
                for environment in pool.environments
            )
            action_counts = np.sum(
                [
                    environment.stepinfo.action_counts
                    for environment in pool.environments
                ],
                axis=0,
            )
            action_trace = {
                env_id: environment.stepinfo.action_trace
                for env_id, environment in enumerate(pool.environments)
            }
            last_info_keys = {
                env_id: environment.stepinfo.last_info_keys
                for env_id, environment in enumerate(pool.environments)
            }
            final_step_info = {
                env_id: environment.stepinfo.final_step_info
                for env_id, environment in enumerate(pool.environments)
            }
            terminated = {
                env_id: environment.stepinfo.terminated
                for env_id, environment in enumerate(pool.environments)
            }
            truncated = {
                env_id: environment.stepinfo.truncated
                for env_id, environment in enumerate(pool.environments)
            }

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
                data["save_every_x_episode"]
            )

            if (
                save_interval > 0
                and episode_number
                % save_interval == 0
            ):

                checkpoint_path = (
                    _save_checkpoint(
                        policy,
                        adapters[0],
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
            adapters[0],
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