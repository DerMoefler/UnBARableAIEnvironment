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
# Creating Adapters, rn just mappo specific
# ---------------------------------------------------------------------------

def create_mappo_adapters(number_of_environments: int, data):
    adapters: list[RMAPPOObservationAdapter] = []
    number_of_agents = int(data["number_of_agents"])
    team_id = int(data["training_team_id"])
    max_enemy_slots = max(
        number_of_agents,
        int(data["number_of_enemies"]),
    )

    for i in range(number_of_environments):
        adapter = RMAPPOObservationAdapter.from_config(
            data,
            team_id=team_id,
            num_agents=number_of_agents,
            max_enemy_slots=max_enemy_slots,
        )
        adapters.append(adapter)

        logger.info(
                    "Created observation adapter: %s"
                    "obs_dim=%d target_feature_dim=%d "
                    "max_enemy_slots=%d",
                    i,
                    adapter.obs_dim,
                    adapter.target_feature_dim,
                    adapter.max_enemy_slots,
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
    *,
    model_name: str,
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

    model_path = Path(
        model_name
    )

    if model_path.name != model_name:
        raise ValueError(
            "model_name must be a file name without "
            f"directories, got {model_name!r}."
        )

    if not model_path.suffix:
        raise ValueError(
            "model_name must have a file extension, "
            f"got {model_name!r}."
        )

    latest_path = (
        model_dir
        / model_path.name
    )

    episode_path = (
        model_dir
        / (
            f"{model_path.stem}"
            f"_episode_{episode:04d}"
            f"{model_path.suffix}"
        )
    )

    temporary_episode_path = (
        episode_path.with_suffix(
            episode_path.suffix + ".tmp"
        )
    )

    torch.save(
        checkpoint,
        temporary_episode_path,
    )

    temporary_episode_path.replace(
        episode_path
    )

    temporary_latest_path = (
        latest_path.with_suffix(
            latest_path.suffix + ".tmp"
        )
    )

    torch.save(
        checkpoint,
        temporary_latest_path,
    )

    temporary_latest_path.replace(
        latest_path
    )

    return episode_path



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config_path: Path | None = None,
) -> None:
    if config_path is None:
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

    model_name = config.get(
        "model_name"
    )

    if (
        not isinstance(model_name, str)
        or not model_name.strip()
    ):
        raise ValueError(
            "'model_name' must be a non-empty string."
        )

    model_name = model_name.strip()

    if Path(model_name).name != model_name:
        raise ValueError(
            "'model_name' must contain only a file name, "
            f"not a directory: {model_name!r}."
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

        unbarable_ai_team_ids = (
            environment.unbarable_ai_team_ids
        )

        if not unbarable_ai_team_ids:
            raise RuntimeError(
                "Environment contains no "
                "UnBARableAI team IDs."
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
                / model_name
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
        # Buffers and trainer
        # ---------------------------------------------------------------

        buffers: dict[
            int,
            RolloutBuffer,
        ] = {}

        for team_id, adapter in adapters.items():
            buffers[team_id] = RolloutBuffer(
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
                "team_id=%d capacity=%d",
                team_id,
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

            initialized_teams: set[int] = {
                current_team_id
            }

            maximum_turns = (
                maximum_steps
                * len(unbarable_ai_team_ids)
            )

            while pool.check_done() == False:

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
                    if device.type == "cuda":
                        torch.cuda.synchronize(
                            device
                        )

                    action_types = [
                        int(action)
                        for action
                        in policy_steps[env_id].actions[:, 0]
                    ]
                    action_types = [
                        int(action)
                        for action
                        in policy_steps[env_id].actions[:, 0]
                    ]

                action_trace_by_team[
                    acting_team_id
                ].append(
                    action_types
                )

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
                        action_counts[action_type] += 1
                    else:
                        logger.warning(
                            "Sampled invalid action %d for team %d; "
                            "expected range [0, %d]",
                            action_type,
                            acting_team_id,
                            adapter.action_dim - 1,
                        )

                if (
                    pending_transitions[
                        acting_team_id
                    ]
                    is not None
                ):
                    raise RuntimeError(
                        "A pending transition already exists "
                        "for the acting team: "
                        f"team_id={acting_team_id}."
                    )

                pending_transitions[
                    acting_team_id
                ] = {
                    "prepared": prepared,
                    "policy_step": policy_step,
                    "rnn_states": (
                        rnn_states.copy()
                    ),
                    "rnn_states_critic": (
                        rnn_states_critic.copy()
                    ),
                }

                # -------------------------------------------------------
                # Convert policy output to BAR actions
                # -------------------------------------------------------

                bar_actions = runner.create_bar_actions(
                    policy_step,
                    prepared,
                    team_id=acting_team_id,
                )
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
                    if not isinstance(next_step_info, dict):
                        raise TypeError(
                            "BAR_Environment.step() must return info as a "
                            f"dictionary, got {type(next_step_info).__name__}."
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
                if not isinstance(next_obs_raw, dict):
                    raise TypeError(
                        "BAR_Environment.step() must return the observation "
                        f"as a dictionary, got {type(next_obs_raw).__name__}."
                    )

                if (
                    "acting_team_id" in next_step_info
                    and int(
                        next_step_info["acting_team_id"]
                    ) != acting_team_id
                ):
                    raise RuntimeError(
                        "Environment returned information for the wrong "
                        "acting team: "
                        f"expected={acting_team_id}, "
                        f"received={next_step_info['acting_team_id']}."
                    )

                reward_team_id = int(
                    next_step_info["reward_team_id"]
                )

                if reward_team_id not in adapters:
                    raise RuntimeError(
                        "Environment returned a reward for an unknown team: "
                        f"reward_team_id={reward_team_id}, "
                        f"available_team_ids={sorted(adapters)}."
                    )

                reward_adapter = adapters[
                    reward_team_id
                ]

                if reward_team_id not in initialized_teams:
                    reward_adapter.reset_episode(
                        next_obs_raw
                    )

                    initialized_teams.add(
                        reward_team_id
                    )

                    logger.debug(
                        "Episode %d initialized team=%d "
                        "stable unit slots=%s",
                        episode_number,
                        reward_team_id,
                        reward_adapter.agent_unit_ids,
                    )

                next_prepared = reward_adapter.prepare(
                    obs_raw=next_obs_raw,
                )

                prepared_by_team[
                    reward_team_id
                ] = next_prepared

                returned_current_team_id = int(
                    next_step_info["current_team_id"]
                )

                if returned_current_team_id != reward_team_id:
                    raise RuntimeError(
                        "Environment returned inconsistent team IDs: "
                        f"current_team_id={returned_current_team_id}, "
                        f"reward_team_id={reward_team_id}."
                    )


                pending = pending_transitions[
                    reward_team_id
                ]

                if pending is not None:
                    reward_array = (
                        reward_adapter.prepare_reward(
                            reward
                        )
                    )

                    masks = np.ones(
                        (
                            reward_adapter.num_agents,
                            1,
                        ),
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

            for team_id in unbarable_ai_team_ids:
                final_enemy_alive = (
                    final_enemy_alive_by_team[
                        team_id
                    ]
                )

                final_ally_alive = (
                    final_ally_alive_by_team[
                        team_id
                    ]
                )

                if (
                    final_enemy_alive is not None
                    and int(final_enemy_alive) <= 0
                ):
                    episode_won_by_team[
                        team_id
                    ] = True

                if (
                    final_ally_alive is not None
                    and int(final_ally_alive) <= 0
                ):
                    episode_lost_by_team[
                        team_id
                    ] = True

            # -----------------------------------------------------------
            # Bootstrap and returns
            # -----------------------------------------------------------

            for team_id in non_empty_team_ids:
                adapter = adapters[team_id]
                buffer = buffers[team_id]

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
                    # Bootstrap aus der letzten verfügbaren
                    # Observation des jeweiligen Teams.
                    if team_id not in prepared_by_team:
                        raise RuntimeError(
                            "No prepared observation is available "
                            "for truncated-episode bootstrapping: "
                            f"team_id={team_id}."
                        )

                    next_value = (
                        runner.predict_values(
                            prepared_by_team[
                                team_id
                            ].share_obs,
                            num_agents=adapter.num_agents,
                        )
                    )

                buffer.compute_returns(
                    next_value,
                    gamma=float(
                        mappo_config["gamma"]
                    ),
                )

            # -----------------------------------------------------------
            # PPO update
            # -----------------------------------------------------------

            trainer.prep_training()
            runner.set_training_mode()

            train_info = trainer.train_multiple(
                buffers=[
                    buffers[team_id]
                    for team_id in non_empty_team_ids
                ],
                update_actor=True,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(
                    device
                )

            for buffer in buffers.values():
                buffer.reset()

            # Ein gemeinsamer Logging-Wert als Mittelwert
            # über alle trainierten Teams.
            value_loss = float(
                train_info.get(
                    "value_loss",
                    0.0,
                )
            )

            policy_loss = float(
                train_info.get(
                    "policy_loss",
                    0.0,
                )
            )

            entropy = float(
                train_info.get(
                    "dist_entropy",
                    0.0,
                )
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
                "Average Reward per Turn: %.4f",
                average_reward_per_step,
            )
            logger.info(
                "Total Episode Reward: %.4f",
                episode_reward,
            )
            logger.info(
                "Player Turns: %d",
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

            for team_id in unbarable_ai_team_ids:
                logger.info(
                    "Team %d Episode Reward: %.4f",
                    team_id,
                    episode_rewards_by_team[
                        team_id
                    ],
                )
                logger.info(
                    "Team %d Won: %s",
                    team_id,
                    episode_won_by_team[
                        team_id
                    ],
                )
                logger.info(
                    "Team %d Lost: %s",
                    team_id,
                    episode_lost_by_team[
                        team_id
                    ],
                )
                logger.info(
                    "Team %d Final Ally Alive: %s",
                    team_id,
                    final_ally_alive_by_team[
                        team_id
                    ],
                )
                logger.info(
                    "Team %d Final Enemy Alive: %s",
                    team_id,
                    final_enemy_alive_by_team[
                        team_id
                    ],
                )
                logger.info(
                    "Team %d Enemy Damage Done: %.4f",
                    team_id,
                    enemy_damage_done_by_team[
                        team_id
                    ],
                )
                logger.info(
                    "Team %d Own Damage Taken: %.4f",
                    team_id,
                    own_damage_taken_by_team[
                        team_id
                    ],
                )
                logger.info(
                    "Team %d Action Counts: %s",
                    team_id,
                    action_counts_by_team[
                        team_id
                    ].tolist(),
                )
                logger.debug(
                    "Team %d Action Trace: %s",
                    team_id,
                    action_trace_by_team[
                        team_id
                    ],
                )
            logger.info(
                "Combined PPO Train Info: %s",
                train_info,
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
                "Combined  Value Loss: %.6f",
                value_loss,
            )
            logger.info(
                "Combined  Policy Loss: %.6f",
                policy_loss,
            )
            logger.info(
                "Combined  Entropy: %.6f",
                entropy,
            )

            # -----------------------------------------------------------
            # Periodic checkpoint
            # -----------------------------------------------------------

            save_interval = int(
                training_config[
                    "save_every_x_episode"
                ]
            )

            if (
                save_interval > 0
                and episode_number
                % save_interval == 0
            ):
                checkpoint_path = (
                    _save_checkpoint(
                        policy,
                        reference_adapters[0],
                        episode_number,
                        run_directory,
                        model_name=model_name,
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
            reference_adapters[0],
            final_episode,
            run_directory,
            model_name=model_name,
        )

        logger.info(
            "Final episode checkpoint saved to: %s",
            final_checkpoint,
        )
        logger.info(
            "Latest model saved to: %s",
            run_directory
            / model_name,
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