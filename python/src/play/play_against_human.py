from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import bar_ai

from src.environment.bar_environment import BAR_Environment
from src.train.mappo.policy import R_MAPPO_Policy
from src.train.mappo.obs_r_mappo import (
    OBS_DIM,
    UNIT_SLOT_FEATURE_DIM,
    build_r_mappo_action_context,
    _prepare_obs,
    _prepare_available_actions,
)


CONFIG_PATH = Path(
    "src/train/experiments/3v3Test/3v3Test.json"
)

CHECKPOINT_PATH = Path(
    "src/train/experiments/3v3Test/run/"
    "r_mappo_latest.pt"
)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    required_keys = {
        "action_dimension",
        "device",
        "learning_rate",
        "number_of_agents",
        "training_team_id",
    }

    missing_keys = sorted(
        required_keys - config.keys()
    )

    if missing_keys:
        raise ValueError(
            "Missing configuration keys: "
            + ", ".join(missing_keys)
        )

    return config


def select_device(config: dict[str, Any]) -> torch.device:
    configured_device = config["device"]

    if configured_device == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(configured_device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was configured, but CUDA is unavailable."
        )

    return device


def load_policy(
    config: dict[str, Any],
    device: torch.device,
) -> R_MAPPO_Policy:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    action_dimension = int(
        config["action_dimension"]
    )

    policy = R_MAPPO_Policy(
        OBS_DIM,
        action_dimension,
        device=device,
        lr=float(config["learning_rate"]),
        target_feature_dim=UNIT_SLOT_FEATURE_DIM,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must contain a dictionary."
        )

    checkpoint_obs_dim = int(
        checkpoint.get("obs_dim", -1)
    )
    checkpoint_action_dim = int(
        checkpoint.get("action_dim", -1)
    )
    checkpoint_num_agents = int(
        checkpoint.get("num_agents", -1)
    )
    checkpoint_team_id = int(
        checkpoint.get("training_team_id", -1)
    )

    expected_num_agents = int(
        config["number_of_agents"]
    )
    expected_team_id = int(
        config["training_team_id"]
    )

    if checkpoint_obs_dim != OBS_DIM:
        raise ValueError(
            "Checkpoint observation dimension does not match: "
            f"checkpoint={checkpoint_obs_dim}, "
            f"expected={OBS_DIM}."
        )

    if checkpoint_action_dim != action_dimension:
        raise ValueError(
            "Checkpoint action dimension does not match: "
            f"checkpoint={checkpoint_action_dim}, "
            f"expected={action_dimension}."
        )

    if checkpoint_num_agents != expected_num_agents:
        raise ValueError(
            "Checkpoint agent count does not match: "
            f"checkpoint={checkpoint_num_agents}, "
            f"expected={expected_num_agents}."
        )

    if checkpoint_team_id != expected_team_id:
        raise ValueError(
            "Checkpoint team ID does not match: "
            f"checkpoint={checkpoint_team_id}, "
            f"expected={expected_team_id}."
        )

    if "actor_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint contains no actor_state_dict."
        )

    policy.actor.load_state_dict(
        checkpoint["actor_state_dict"]
    )

    # Der Critic wird für das reine Spielen nicht benötigt.
    # Laden ist trotzdem sinnvoll, wenn er vorhanden ist.
    if "critic_state_dict" in checkpoint:
        policy.critic.load_state_dict(
            checkpoint["critic_state_dict"]
        )

    policy.actor.eval()
    policy.critic.eval()

    logging.info(
        "Loaded checkpoint %s from episode %s",
        CHECKPOINT_PATH,
        checkpoint.get("episode"),
    )

    return policy


def make_bar_action(
    action_index: Any,
    unit_id: Any,
    target_unit_id: Any,
    training_team_id: int,
) -> bar_ai.Action:
    action_value = int(action_index) + 1
    unit_value = int(unit_id)
    target_value = int(target_unit_id)

    action_id = (
        bar_ai.ActionId(action_value)
        if hasattr(bar_ai, "ActionId")
        else action_value
    )

    action = bar_ai.Action()
    action.action_id = action_id
    action.unit_id = unit_value
    action.target_unit_id = target_value

    if hasattr(action, "team_id"):
        action.team_id = int(training_team_id)

    if hasattr(action, "ally_team_id"):
        action.ally_team_id = int(training_team_id)

    return action


def sample_actions(
    policy: R_MAPPO_Policy,
    obs_raw: Any,
    obs: np.ndarray,
    available_actions: np.ndarray,
    training_team_id: int,
    num_agents: int,
    device: torch.device,
) -> np.ndarray:
    (
        own_unit_ids,
        candidate_target_ids,
        target_features,
        target_mask,
    ) = build_r_mappo_action_context(
        obs_raw,
        training_team_id=training_team_id,
    )

    if len(own_unit_ids) != num_agents:
        raise RuntimeError(
            "Unexpected number of controlled units: "
            f"observed={len(own_unit_ids)}, "
            f"expected={num_agents}."
        )

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

    target_features_tensor = torch.as_tensor(
        target_features,
        dtype=torch.float32,
        device=device,
    )

    target_mask_tensor = torch.as_tensor(
        target_mask,
        dtype=torch.bool,
        device=device,
    )

    with torch.inference_mode():
        actions, _ = policy.get_action(
            obs_tensor,
            target_features=target_features_tensor,
            candidate_target_ids=candidate_target_ids,
            target_mask=target_mask_tensor,
            available_actions=available_actions_tensor,
        )

    if isinstance(actions, torch.Tensor):
        action_array = (
            actions.detach().cpu().numpy()
        )
    else:
        action_array = np.asarray(actions)

    return action_array.reshape(
        num_agents,
        2,
    )


def convert_actions(
    policy: R_MAPPO_Policy,
    policy_actions: np.ndarray,
    obs_raw: Any,
    training_team_id: int,
) -> list[bar_ai.Action]:
    (
        own_unit_ids,
        candidate_target_ids,
        _,
        candidate_target_mask,
    ) = build_r_mappo_action_context(
        obs_raw,
        training_team_id=training_team_id,
    )

    if len(own_unit_ids) != len(policy_actions):
        raise RuntimeError(
            "Policy action count does not match controlled units: "
            f"actions={len(policy_actions)}, "
            f"units={len(own_unit_ids)}."
        )

    bar_actions: list[bar_ai.Action] = []

    for row, policy_action in enumerate(policy_actions):
        action_type = int(policy_action[0])
        target_index = int(policy_action[1])

        if action_type == policy.attack_action_index:
            if not (
                0
                <= target_index
                < candidate_target_ids.shape[1]
            ):
                raise ValueError(
                    f"Invalid target index {target_index} "
                    f"for unit {own_unit_ids[row]}."
                )

            if not candidate_target_mask[
                row,
                target_index,
            ]:
                raise ValueError(
                    f"Target slot {target_index} is empty "
                    f"for unit {own_unit_ids[row]}."
                )

            target_unit_id = int(
                candidate_target_ids[
                    row,
                    target_index,
                ]
            )
        else:
            # Wird für Nicht-Angriffsaktionen vom Plugin ignoriert.
            target_unit_id = 0

        bar_actions.append(
            make_bar_action(
                action_index=action_type,
                unit_id=own_unit_ids[row],
                target_unit_id=target_unit_id,
                training_team_id=training_team_id,
            )
        )

    if not bar_actions:
        raise RuntimeError(
            "No actions were generated for the AI team."
        )

    return bar_actions


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
    )

    config = load_config()
    device = select_device(config)

    training_team_id = int(
        config["training_team_id"]
    )
    num_agents = int(
        config["number_of_agents"]
    )
    action_dimension = int(
        config["action_dimension"]
    )

    policy = load_policy(
        config=config,
        device=device,
    )

    environment = BAR_Environment(
        training_team_id=training_team_id,

        # Hoch genug für einen ersten menschlichen Test.
        max_episode_steps=100_000,

        engine_id=0,
        port=50051,
    )

    try:
        logging.info(
            "Starting human-vs-AI match on device %s",
            device,
        )

        obs_raw, reset_info = environment.reset()

        obs = _prepare_obs(
            obs_raw,
            num_agents,
            OBS_DIM,
            training_team_id=training_team_id,
        )

        available_actions = _prepare_available_actions(
            None,
            num_agents,
            action_dimension,
        )

        terminated = False
        truncated = False
        step_count = 0
        final_info: dict[str, Any] = reset_info

        while not (terminated or truncated):
            policy_actions = sample_actions(
                policy=policy,
                obs_raw=obs_raw,
                obs=obs,
                available_actions=available_actions,
                training_team_id=training_team_id,
                num_agents=num_agents,
                device=device,
            )

            bar_actions = convert_actions(
                policy=policy,
                policy_actions=policy_actions,
                obs_raw=obs_raw,
                training_team_id=training_team_id,
            )

            (
                next_obs_raw,
                reward,
                terminated,
                truncated,
                step_info,
            ) = environment.step(bar_actions)

            step_count += 1
            final_info = step_info

            logging.debug(
                "step=%d reward=%.4f "
                "terminated=%s truncated=%s actions=%s",
                step_count,
                float(reward),
                terminated,
                truncated,
                policy_actions.tolist(),
            )

            if terminated or truncated:
                break

            obs_raw = next_obs_raw

            obs = _prepare_obs(
                obs_raw,
                num_agents,
                OBS_DIM,
                training_team_id=training_team_id,
            )

            available_actions = _prepare_available_actions(
                None,
                num_agents,
                action_dimension,
            )

        logging.info(
            "Match finished: steps=%d terminated=%s "
            "truncated=%s outcome=%s",
            step_count,
            terminated,
            truncated,
            final_info.get("episode_outcome"),
        )

        print(
            "Match beendet:",
            f"outcome={final_info.get('episode_outcome')},",
            f"steps={step_count},",
            f"terminated={terminated},",
            f"truncated={truncated}",
            flush=True,
        )

    except KeyboardInterrupt:
        logging.info(
            "Human-vs-AI match interrupted by user."
        )

    finally:
        environment.close()


if __name__ == "__main__":
    main()