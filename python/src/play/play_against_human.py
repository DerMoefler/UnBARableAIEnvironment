from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.environment.bar_environment import BAR_Environment
from src.environment.episode_outcome import EpisodeOutcome
from src.train.mappo.obs_r_mappo import (
    RMAPPOObservationAdapter,
)
from src.train.mappo.policy_runner import (
    MAPPOPolicyRunner,
)


CONFIG_PATH = Path(
    "src/play/config/3v3Test.json"
)

MODELS_PATH = Path(
    "src/train/experiments/3v3Test/run/"
)


def load_config(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must be a JSON object."
        )

    return config


def select_device(
    config: dict[str, Any],
) -> torch.device:
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
    )

    config = load_config(
        CONFIG_PATH
    )

    if (
        config.get(
            "engine",
            {},
        ).get(
            "mode"
        )
        != "play_against_human"
    ):
        raise ValueError(
            "Set engine.mode to "
            "'play_against_human' in the configuration."
        )

    checkpoint_path = (
        MODELS_PATH
        / config["model_name"]
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{checkpoint_path}"
        )

    device = select_device(
        config
    )

    # -----------------------------------------------------------
    # Environment
    # -----------------------------------------------------------

    environment = (
        BAR_Environment.from_config(
            CONFIG_PATH,
            environment_id=0,
            max_episode_steps=10000,
        )
    )

    if len(
        environment.unbarable_ai_team_ids
    ) != 1:
        environment.close()

        raise ValueError(
            "play_against_human requires exactly one "
            "UnBARableAI team, but the environment "
            f"configured these teams: "
            f"{environment.unbarable_ai_team_ids}"
        )

    controlled_team_id = (
        environment.unbarable_ai_team_ids[0]
    )

    number_of_agents = int(
        config["number_of_agents"]
    )

    number_of_enemies = int(
        config["number_of_enemies"]
    )

    # Muss mit dem Training und den Checkpoint-
    # Dimensionen übereinstimmen.
    max_enemy_slots = max(
        number_of_agents,
        number_of_enemies,
    )

    # -----------------------------------------------------------
    # Observation adapter and policy
    # -----------------------------------------------------------

    adapter = (
        RMAPPOObservationAdapter.from_config(
            config,
            team_id=controlled_team_id,
            num_agents=number_of_agents,
            max_enemy_slots=max_enemy_slots,
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

    loaded_episode = (
        runner.load_checkpoint(
            checkpoint_path,
            load_optimizers=False,
        )
    )

    runner.set_evaluation_mode()

    logging.info(
        "Loaded checkpoint from episode %d",
        loaded_episode,
    )

    logging.info(
        "Playing with UnBARableAI team %d: "
        "num_agents=%d max_enemy_slots=%d "
        "obs_dim=%d device=%s",
        controlled_team_id,
        adapter.num_agents,
        adapter.max_enemy_slots,
        adapter.obs_dim,
        device,
    )

    try:
        logging.info(
            "Starting human-vs-AI match"
        )

        (
            obs_raw,
            reset_info,
        ) = environment.reset()

        current_team_id = int(
            reset_info["current_team_id"]
        )

        if (
            current_team_id
            != controlled_team_id
        ):
            raise RuntimeError(
                "Environment returned the wrong "
                "UnBARableAI team at reset: "
                f"expected={controlled_team_id}, "
                f"received={current_team_id}."
            )

        adapter.reset_episode(
            obs_raw
        )

        prepared = adapter.prepare(
            obs_raw=obs_raw,
        )

        terminated = False
        truncated = False
        step_count = 0

        final_info: dict[
            str,
            Any,
        ] = reset_info

        while not (
            terminated or truncated
        ):
            policy_step = runner.act(
                prepared
            )

            bar_actions = (
                runner.create_bar_actions(
                    policy_step,
                    prepared,
                    team_id=current_team_id,
                )
            )

            (
                next_obs_raw,
                reward,
                terminated,
                truncated,
                step_info,
            ) = environment.step(
                bar_actions
            )

            step_count += 1
            final_info = step_info

            logging.debug(
                "step=%d team_id=%d reward=%s "
                "terminated=%s truncated=%s",
                step_count,
                current_team_id,
                reward,
                terminated,
                truncated,
            )

            if (
                terminated
                or truncated
            ):
                break

            if not isinstance(
                next_obs_raw,
                dict,
            ):
                raise RuntimeError(
                    "Environment returned no valid "
                    "observation while the match "
                    "is still running."
                )

            current_team_id = int(
                step_info[
                    "current_team_id"
                ]
            )

            if (
                current_team_id
                != controlled_team_id
            ):
                raise RuntimeError(
                    "Environment switched to an "
                    "unexpected UnBARableAI team: "
                    f"expected="
                    f"{controlled_team_id}, "
                    f"received="
                    f"{current_team_id}."
                )

            prepared = adapter.prepare(
                obs_raw=next_obs_raw,
            )

        logging.info(
            "Match finished: steps=%d "
            "terminated=%s truncated=%s "
            "outcome=%s",
            step_count,
            terminated,
            truncated,
            final_info.get(
                "episode_outcome"
            ),
        )

        outcome = final_info.get(
            "episode_outcome"
        )

        if (
            terminated
            and outcome
            == EpisodeOutcome.WON.value
        ):
            print(
                "You lost the match against "
                "the AI. Better luck next time!"
            )

        elif (
            terminated
            and outcome
            == EpisodeOutcome.LOST.value
        ):
            print(
                "Congratulations! You won "
                "the match against the AI!"
            )

        elif truncated:
            print(
                "The match was truncated."
            )

        else:
            print(
                "The match ended without a "
                "recognized outcome."
            )

    except KeyboardInterrupt:
        logging.info(
            "Human-vs-AI match interrupted "
            "by user."
        )

    finally:
        environment.close()


if __name__ == "__main__":
    main()