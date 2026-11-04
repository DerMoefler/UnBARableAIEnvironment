from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import bar_ai

from src.train.mappo.obs_r_mappo import (
    OBS_DIM,
    UNIT_SLOT_FEATURE_DIM,
    build_r_mappo_action_context,
    _prepare_available_actions,
    _prepare_obs,
    _prepare_share_obs,
)
from src.train.mappo.policy import R_MAPPO_Policy


class MAPPOPolicyRunner:
    """
    Runs an R-MAPPO policy against a BAR environment.

    This class is shared by training, evaluation and human-vs-AI play.
    It does not train the policy.
    """

    def __init__(
        self,
        policy: R_MAPPO_Policy,
        *,
        device: torch.device,
        num_agents: int,
        action_dimension: int,
        training_team_id: int,
    ) -> None:
        self.policy = policy
        self.device = device
        self.num_agents = int(num_agents)
        self.action_dimension = int(action_dimension)
        self.training_team_id = int(training_team_id)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        device: torch.device,
    ) -> "MAPPOPolicyRunner":
        policy = R_MAPPO_Policy(
            OBS_DIM,
            int(config["action_dimension"]),
            device=device,
            lr=float(config["learning_rate"]),
            target_feature_dim=UNIT_SLOT_FEATURE_DIM,
        )

        return cls(
            policy=policy,
            device=device,
            num_agents=int(config["number_of_agents"]),
            action_dimension=int(config["action_dimension"]),
            training_team_id=int(config["training_team_id"]),
        )

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        load_optimizers: bool = False,
    ) -> int:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Checkpoint must contain a dictionary: "
                f"{checkpoint_path}"
            )

        self._validate_checkpoint(
            checkpoint,
            checkpoint_path,
        )

        self.policy.actor.load_state_dict(
            checkpoint["actor_state_dict"]
        )
        self.policy.critic.load_state_dict(
            checkpoint["critic_state_dict"]
        )

        if load_optimizers:
            actor_optimizer_state = checkpoint.get(
                "actor_optimizer_state_dict"
            )
            if actor_optimizer_state is not None:
                self.policy.actor_optimizer.load_state_dict(
                    actor_optimizer_state
                )

            critic_optimizer_state = checkpoint.get(
                "critic_optimizer_state_dict"
            )
            if critic_optimizer_state is not None:
                self.policy.critic_optimizer.load_state_dict(
                    critic_optimizer_state
                )

        return int(checkpoint.get("episode", 0))

    def prepare_observations(
        self,
        obs_raw: Any,
        share_obs_raw: Any = None,
        available_actions_raw: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = _prepare_obs(
            obs_raw,
            self.num_agents,
            OBS_DIM,
            training_team_id=self.training_team_id,
        )

        share_obs = _prepare_share_obs(
            share_obs_raw,
            obs,
            self.num_agents,
            OBS_DIM,
        )

        available_actions = _prepare_available_actions(
            available_actions_raw,
            self.num_agents,
            self.action_dimension,
        )

        return obs, share_obs, available_actions

    def sample_actions(
        self,
        *,
        obs_raw: Any,
        obs: np.ndarray,
        share_obs: np.ndarray,
        available_actions: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        (
            own_unit_ids,
            candidate_target_ids,
            target_features,
            target_mask,
        ) = build_r_mappo_action_context(
            obs_raw,
            training_team_id=self.training_team_id,
        )

        if len(own_unit_ids) != self.num_agents:
            raise ValueError(
                "Number of controlled units does not match "
                f"num_agents: observed={len(own_unit_ids)}, "
                f"expected={self.num_agents}."
            )

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        )

        available_actions_tensor = torch.as_tensor(
            available_actions,
            dtype=torch.bool,
            device=self.device,
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
            device=self.device,
        )
        target_mask_tensor = torch.as_tensor(
            target_mask_array,
            dtype=torch.bool,
            device=self.device,
        )

        actions, action_log_probs = self.policy.get_action(
            obs_tensor,
            target_features=target_features_tensor,
            candidate_target_ids=candidate_target_ids,
            target_mask=target_mask_tensor,
            available_actions=available_actions_tensor,
        )

        action_array = self._to_numpy(actions).reshape(
            self.num_agents,
            2,
        ).astype(np.float32)

        log_probability_array = self._to_numpy(
            action_log_probs
        ).reshape(
            self.num_agents,
            1,
        ).astype(np.float32)

        with torch.no_grad():
            share_obs_tensor = torch.as_tensor(
                share_obs,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            values = self.policy.critic(
                share_obs_tensor
            )

        value_predictions = self._repeat_values(
            values
        )

        return (
            value_predictions,
            action_array,
            log_probability_array,
            target_features_array,
            target_mask_array,
        )

    def step_environment(
        self,
        env: Any,
        policy_actions: np.ndarray,
        obs_raw: Any,
    ) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        action_rows = np.asarray(policy_actions)

        if action_rows.ndim == 1:
            action_rows = action_rows.reshape(1, -1)

        if (
            action_rows.ndim != 2
            or action_rows.shape[1] != 2
        ):
            raise ValueError(
                "Policy actions must have shape "
                f"(num_agents, 2), got {action_rows.shape}."
            )

        (
            own_unit_ids,
            candidate_target_ids,
            _,
            candidate_target_mask,
        ) = build_r_mappo_action_context(
            obs_raw,
            training_team_id=self.training_team_id,
        )

        if len(own_unit_ids) != len(action_rows):
            raise ValueError(
                f"Got {len(action_rows)} actions for "
                f"{len(own_unit_ids)} controlled units."
            )

        bar_actions: list[bar_ai.Action] = []

        for row, (
            action_type,
            target_index,
        ) in enumerate(action_rows):
            action_type = int(action_type)
            target_index = int(target_index)

            if (
                action_type
                == self.policy.attack_action_index
            ):
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
                target_unit_id = 4211

            bar_actions.append(
                self._make_bar_action(
                    action_index=action_type,
                    unit_id=own_unit_ids[row],
                    target_unit_id=target_unit_id,
                )
            )

        if not bar_actions:
            raise ValueError(
                "At least one agent action is required."
            )

        result = env.step(bar_actions)

        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError(
                "BAR_Environment.step must return "
                "(obs, reward, terminated, truncated, info)."
            )

        return result

    def _make_bar_action(
        self,
        *,
        action_index: Any,
        unit_id: Any,
        target_unit_id: Any,
    ) -> bar_ai.Action:
        action_value = int(action_index) + 1
        unit_value = int(unit_id)
        target_value = int(target_unit_id)

        action_id = (
            bar_ai.ActionId(action_value)
            if hasattr(bar_ai, "ActionId")
            else action_value
        )

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

        try:
            return bar_ai.Action(
                action_id,
                unit_value,
                target_value,
            )
        except TypeError:
            pass

        try:
            action = bar_ai.Action(
                action_id,
                unit_value,
            )
        except TypeError:
            action = bar_ai.Action()

        field_options = (
            (
                ("action_id", "actionId"),
                action_id,
            ),
            (
                ("unit_id", "unitId"),
                unit_value,
            ),
            (
                ("target_unit_id", "targetUnitId"),
                target_value,
            ),
        )

        for field_names, value in field_options:
            field_name = next(
                (
                    name
                    for name in field_names
                    if hasattr(action, name)
                ),
                None,
            )

            if field_name is None:
                raise TypeError(
                    "bar_ai.Action does not expose any of "
                    f"these fields: {field_names}."
                )

            setattr(action, field_name, value)

        return action

    def _validate_checkpoint(
        self,
        checkpoint: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        expected = (
            OBS_DIM,
            self.action_dimension,
            self.num_agents,
            self.training_team_id,
        )

        actual = (
            int(checkpoint.get("obs_dim", -1)),
            int(checkpoint.get("action_dim", -1)),
            int(checkpoint.get("num_agents", -1)),
            int(checkpoint.get("training_team_id", -1)),
        )

        if actual != expected:
            raise ValueError(
                "Checkpoint configuration does not match: "
                f"checkpoint={actual}, expected={expected}, "
                f"path={checkpoint_path}."
            )

        required_states = (
            "actor_state_dict",
            "critic_state_dict",
        )

        missing_states = [
            key
            for key in required_states
            if key not in checkpoint
        ]

        if missing_states:
            raise ValueError(
                "Checkpoint is missing required states: "
                + ", ".join(missing_states)
            )

    def _repeat_values(
        self,
        value_tensor: torch.Tensor,
    ) -> np.ndarray:
        values = (
            value_tensor
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        if values.size == 1:
            return np.full(
                (self.num_agents, 1),
                float(values[0]),
                dtype=np.float32,
            )

        result = np.zeros(
            self.num_agents,
            dtype=np.float32,
        )

        count = min(
            self.num_agents,
            values.size,
        )
        result[:count] = values[:count]

        return result.reshape(
            self.num_agents,
            1,
        )

    def predict_values(
        self,
        share_obs: np.ndarray,
    ) -> np.ndarray:
        with torch.no_grad():
            share_obs_tensor = torch.as_tensor(
                share_obs,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            values = self.policy.critic(
                share_obs_tensor
            )

        return self._repeat_values(values)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()

        return np.asarray(value)