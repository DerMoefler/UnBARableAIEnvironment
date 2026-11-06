from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.environment.bar_environment import BAR_Environment
from src.train.mappo.obs_r_mappo import (
    RMAPPOObservationAdapter,
)
from src.train.mappo.policy_runner import (
    MAPPOPolicyRunner,
)


CONFIG_PATH = Path(
    "src/play/config/3v3Test.json"
)

CHECKPOINT_PATH = Path(
    "src/train/experiments/3v3Test/run/"
    "r_mappo_latest.pt"
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

    config = load_config(CONFIG_PATH)

    if (
        config.get("engine", {}).get("mode")
        != "play_against_human"
    ):
        raise ValueError(
            "Set engine.mode to "
            "'play_against_human' in the configuration."
        )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = select_device(config)

    adapter = RMAPPOObservationAdapter.from_config(
        config
    )

    runner = MAPPOPolicyRunner.from_config(
        config,
        device,
        obs_dim=adapter.obs_dim,
        target_feature_dim=(
            adapter.target_feature_dim
        ),
    )

    loaded_episode = runner.load_checkpoint(
        CHECKPOINT_PATH,
        load_optimizers=False,
    )

    runner.set_evaluation_mode()

    logging.info(
        "Loaded checkpoint from episode %d",
        loaded_episode,
    )

    environment = BAR_Environment(
        training_team_id=(
            adapter.training_team_id
        ),
        max_episode_steps=100_000,
        config_path=CONFIG_PATH,
        environment_id=0,
    )

    try:
        logging.info(
            "Starting human-vs-AI match on %s",
            device,
        )

        obs_raw, reset_info = environment.reset()

        adapter.reset_episode(obs_raw)

        prepared = adapter.prepare(
            obs_raw=obs_raw,
        )

        terminated = False
        truncated = False
        step_count = 0

        final_info: dict[str, Any] = reset_info

        while not (terminated or truncated):
            policy_step = runner.act(
                prepared
            )

            bar_actions = (
                runner.create_bar_actions(
                    policy_step,
                    prepared,
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
                "step=%d reward=%s "
                "terminated=%s truncated=%s",
                step_count,
                reward,
                terminated,
                truncated,
            )

            if not (terminated or truncated):
                if next_obs_raw is None:
                    raise RuntimeError(
                        "Environment returned no observation "
                        "while the match is still running."
                    )

                prepared = adapter.prepare(
                    obs_raw=next_obs_raw,
                )

        logging.info(
            "Match finished: steps=%d "
            "terminated=%s truncated=%s outcome=%s",
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