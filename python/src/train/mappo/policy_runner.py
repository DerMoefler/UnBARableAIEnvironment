from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import bar_ai

from src.train.mappo.obs_r_mappo import (
    PreparedStep,
)
from src.train.mappo.policy import (
    R_MAPPO_Policy,
)


MOVEMENT_TARGET_ID = 4211


@dataclass(frozen=True)
class PolicyStep:
    """
    Output produced by one R-MAPPO policy inference step.
    """

    value_predictions: np.ndarray
    actions: np.ndarray
    action_log_probs: np.ndarray


class MAPPOPolicyRunner:
    """
    Runs an R-MAPPO policy on already prepared observations.

    The runner does not interpret raw BAR observations and does not
    manage controlled-unit slots. Those responsibilities belong to
    RMAPPOObservationAdapter.

    Responsibilities:

    - construct the R-MAPPO policy,
    - load checkpoints (from other models),
    - run actor and critic inference,
    - convert sampled policy actions into bar_ai.Action objects.
    """

    def __init__(
        self,
        policy: R_MAPPO_Policy,
        *,
        device: torch.device,
        num_agents: int,
        action_dim: int,
        obs_dim: int,
        target_feature_dim: int,
        training_team_id: int,
    ) -> None:
        self.policy = policy
        self.device = torch.device(device)

        self.num_agents = int(num_agents)
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.target_feature_dim = int(
            target_feature_dim
        )
        self.training_team_id = int(
            training_team_id
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        device: torch.device,
        *,
        obs_dim: int,
        target_feature_dim: int,
    ) -> MAPPOPolicyRunner:
        training_config = config.get("training", {})
        mappo_config = training_config.get("mappo", {})

        num_agents = int(
            config["number_of_agents"]
        )
        action_dim = int(
            config["action_dimension"]
        )
        training_team_id = int(
            config["training_team_id"]
        )

        learing_rate = float(
            mappo_config.get("learning_rate", 0.0001)
        )

        policy = R_MAPPO_Policy(
            obs_dim,
            action_dim,
            device=device,
            lr=learing_rate,
            target_feature_dim=target_feature_dim,
        )

        return cls(
            policy=policy,
            device=device,
            num_agents=num_agents,
            action_dim=action_dim,
            obs_dim=obs_dim,
            target_feature_dim=target_feature_dim,
            training_team_id=training_team_id,
        )

    def act(
        self,
        prepared: PreparedStep,
    ) -> PolicyStep:
        """
        Run actor, target scorer and critic for one prepared step.
        """
        self._validate_prepared_step(
            prepared
        )

        obs_tensor = torch.as_tensor(
            prepared.obs,
            dtype=torch.float32,
            device=self.device,
        )

        available_actions_tensor = (
            torch.as_tensor(
                prepared.available_actions,
                dtype=torch.bool,
                device=self.device,
            )
        )

        target_features_tensor = (
            torch.as_tensor(
                prepared.target_features,
                dtype=torch.float32,
                device=self.device,
            )
        )

        target_mask_tensor = torch.as_tensor(
            prepared.target_mask,
            dtype=torch.bool,
            device=self.device,
        )

        with torch.no_grad():
            (
                actions,
                action_log_probs,
            ) = self.policy.get_action(
                obs_tensor,
                target_features=(
                    target_features_tensor
                ),
                candidate_target_ids=(
                    prepared.target_ids
                ),
                target_mask=target_mask_tensor,
                available_actions=(
                    available_actions_tensor
                ),
            )

        action_array = self._to_numpy(
            actions
        ).reshape(
            self.num_agents,
            2,
        ).astype(
            np.float32,
            copy=False,
        )

        action_log_prob_array = (
            self._to_numpy(
                action_log_probs
            ).reshape(
                self.num_agents,
                1,
            ).astype(
                np.float32,
                copy=False,
            )
        )

        value_predictions = self.predict_values(
            prepared.share_obs
        )

        return PolicyStep(
            value_predictions=(
                value_predictions
            ),
            actions=action_array,
            action_log_probs=(
                action_log_prob_array
            ),
        )

    def create_bar_actions(
        self,
        policy_step: PolicyStep,
        prepared: PreparedStep,
    ) -> list[bar_ai.Action]:
        """
        Convert policy actions into actions for living BAR units.

        Policy output contains one row for every stable agent slot.
        Rows with own_unit_id = -1 belong to dead or missing units and
        are not sent to BAR.
        """
        self._validate_prepared_step(
            prepared
        )
        self._validate_policy_step(
            policy_step
        )

        actions: list[bar_ai.Action] = []

        for row in range(self.num_agents):
            unit_id = int(
                prepared.own_unit_ids[row]
            )

            # Dead or missing controlled unit.
            if unit_id < 0:
                continue

            action_type = int(
                policy_step.actions[row, 0]
            )
            target_index = int(
                policy_step.actions[row, 1]
            )

            if (
                action_type
                == self.policy.attack_action_index
            ):
                if not (
                    0
                    <= target_index
                    < prepared.target_ids.shape[1]
                ):
                    raise ValueError(
                        "Invalid target index "
                        f"{target_index} for controlled "
                        f"unit {unit_id}."
                    )

                if not bool(
                    prepared.target_mask[
                        row,
                        target_index,
                    ]
                ):
                    raise ValueError(
                        f"Target slot {target_index} "
                        f"is invalid for controlled "
                        f"unit {unit_id}."
                    )

                target_unit_id = int(
                    prepared.target_ids[
                        row,
                        target_index,
                    ]
                )

                if target_unit_id < 0:
                    raise ValueError(
                        "Target mask marked an invalid "
                        f"target ID for unit {unit_id}: "
                        f"{target_unit_id}."
                    )
            else:
                target_unit_id = (
                    MOVEMENT_TARGET_ID
                )

            actions.append(
                self._make_bar_action(
                    action_index=action_type,
                    unit_id=unit_id,
                    target_unit_id=(
                        target_unit_id
                    ),
                )
            )

        if not actions:
            raise RuntimeError(
                "No living controlled units are available, "
                "but an environment step was requested."
            )

        return actions

    def predict_values(
        self,
        share_obs: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the centralized critic value and repeat it per agent.

        Result shape: (num_agents, 1)
        """
        share_obs_array = np.asarray(
            share_obs,
            dtype=np.float32,
        )

        if share_obs_array.shape != (
            self.obs_dim,
        ):
            raise ValueError(
                "Invalid shared observation shape: "
                f"expected {(self.obs_dim,)}, "
                f"got {share_obs_array.shape}."
            )

        share_obs_tensor = torch.as_tensor(
            share_obs_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            values = self.policy.critic(
                share_obs_tensor
            )

        return self._repeat_values(values)

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        load_optimizers: bool = False,
    ) -> int:
        """
        Load actor, critic and optional optimizer states.

        Returns the episode stored in the checkpoint.
        """
        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

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

        # Neuere Checkpoints sollten den TargetScorer separat
        # speichern. Ältere Checkpoints bleiben weiterhin ladbar.
        target_scorer_state = checkpoint.get(
            "target_scorer_state_dict"
        )

        if target_scorer_state is not None:
            self.policy.target_scorer.load_state_dict(
                target_scorer_state
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

        return int(
            checkpoint.get("episode", 0)
        )

    def set_training_mode(self) -> None:
        """
        Enable training mode for actor, target scorer and critic.
        """
        self.policy.actor.train()
        self.policy.target_scorer.train()
        self.policy.critic.train()

    def set_evaluation_mode(self) -> None:
        """
        Enable evaluation mode for actor, target scorer and critic.
        """
        self.policy.actor.eval()
        self.policy.target_scorer.eval()
        self.policy.critic.eval()

    def _make_bar_action(
        self,
        *,
        action_index: Any,
        unit_id: Any,
        target_unit_id: Any,
    ) -> bar_ai.Action:
        """
        Convert one policy action into a bar_ai.Action.
        """
        action_value = int(action_index) + 1
        unit_value = int(unit_id)
        target_value = int(target_unit_id)

        action_id = (
            bar_ai.ActionId(action_value)
            if hasattr(bar_ai, "ActionId")
            else action_value
        )

        constructor_variants = (
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
        )

        for kwargs in constructor_variants:
            try:
                action = bar_ai.Action(**kwargs)
                self._set_optional_team_fields(
                    action
                )
                return action
            except TypeError:
                pass

        try:
            action = bar_ai.Action(
                action_id,
                unit_value,
                target_value,
            )

            self._set_optional_team_fields(
                action
            )

            return action

        except TypeError:
            pass

        try:
            action = bar_ai.Action(
                action_id,
                unit_value,
            )
        except TypeError:
            try:
                action = bar_ai.Action()
            except TypeError as exc:
                raise TypeError(
                    "Could not construct bar_ai.Action."
                ) from exc

        required_fields = (
            (
                ("action_id", "actionId"),
                action_id,
            ),
            (
                ("unit_id", "unitId"),
                unit_value,
            ),
            (
                (
                    "target_unit_id",
                    "targetUnitId",
                ),
                target_value,
            ),
        )

        for field_names, value in required_fields:
            self._set_required_field(
                action,
                field_names,
                value,
            )

        self._set_optional_team_fields(
            action
        )

        return action

    def _set_optional_team_fields(
        self,
        action: bar_ai.Action,
    ) -> None:
        self._set_optional_field(
            action,
            ("team_id", "teamId"),
            self.training_team_id,
        )

        # Falls BAR getrennte AllyTeam-IDs verwendet, sollte dies
        # später ein eigener Konfigurationswert werden.
        self._set_optional_field(
            action,
            (
                "ally_team_id",
                "allyTeamId",
            ),
            self.training_team_id,
        )

    @staticmethod
    def _set_required_field(
        action: bar_ai.Action,
        field_names: tuple[str, ...],
        value: Any,
    ) -> None:
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
                "bar_ai.Action does not expose any "
                f"of these fields: {field_names}."
            )

        try:
            setattr(
                action,
                field_name,
                value,
            )
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "Could not set bar_ai.Action field "
                f"{field_name!r}."
            ) from exc

    @staticmethod
    def _set_optional_field(
        action: bar_ai.Action,
        field_names: tuple[str, ...],
        value: Any,
    ) -> None:
        field_name = next(
            (
                name
                for name in field_names
                if hasattr(action, name)
            ),
            None,
        )

        if field_name is not None:
            setattr(
                action,
                field_name,
                value,
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
            (self.num_agents,),
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

    def _validate_prepared_step(
        self,
        prepared: PreparedStep,
    ) -> None:
        expected_shapes = {
            "obs": (
                self.num_agents,
                self.obs_dim,
            ),
            "share_obs": (
                self.obs_dim,
            ),
            "available_actions": (
                self.num_agents,
                self.action_dim,
            ),
            "own_unit_ids": (
                self.num_agents,
            ),
            "target_ids": (
                self.num_agents,
                prepared.target_ids.shape[1],
            ),
            "target_features": (
                self.num_agents,
                prepared.target_ids.shape[1],
                self.target_feature_dim,
            ),
            "target_mask": (
                self.num_agents,
                prepared.target_ids.shape[1],
            ),
            "active_mask": (
                self.num_agents,
                1,
            ),
        }

        values = {
            "obs": prepared.obs,
            "share_obs": prepared.share_obs,
            "available_actions": (
                prepared.available_actions
            ),
            "own_unit_ids": (
                prepared.own_unit_ids
            ),
            "target_ids": prepared.target_ids,
            "target_features": (
                prepared.target_features
            ),
            "target_mask": prepared.target_mask,
            "active_mask": prepared.active_mask,
        }

        for name, expected_shape in (
            expected_shapes.items()
        ):
            actual_shape = np.asarray(
                values[name]
            ).shape

            if actual_shape != expected_shape:
                raise ValueError(
                    f"PreparedStep.{name} has invalid "
                    f"shape: expected={expected_shape}, "
                    f"actual={actual_shape}."
                )

        expected_active_mask = (
            prepared.own_unit_ids >= 0
        ).astype(
            np.float32
        ).reshape(
            self.num_agents,
            1,
        )

        if not np.array_equal(
            prepared.active_mask,
            expected_active_mask,
        ):
            raise ValueError(
                "PreparedStep.active_mask does not "
                "match own_unit_ids."
            )

    def _validate_policy_step(
        self,
        policy_step: PolicyStep,
    ) -> None:
        expected_shapes = {
            "value_predictions": (
                self.num_agents,
                1,
            ),
            "actions": (
                self.num_agents,
                2,
            ),
            "action_log_probs": (
                self.num_agents,
                1,
            ),
        }

        values = {
            "value_predictions": (
                policy_step.value_predictions
            ),
            "actions": policy_step.actions,
            "action_log_probs": (
                policy_step.action_log_probs
            ),
        }

        for name, expected_shape in (
            expected_shapes.items()
        ):
            actual_shape = np.asarray(
                values[name]
            ).shape

            if actual_shape != expected_shape:
                raise ValueError(
                    f"PolicyStep.{name} has invalid "
                    f"shape: expected={expected_shape}, "
                    f"actual={actual_shape}."
                )

    def _validate_checkpoint(
        self,
        checkpoint: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        expected_metadata = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,
            "training_team_id": (
                self.training_team_id
            ),
        }

        for key, expected_value in (
            expected_metadata.items()
        ):
            actual_value = int(
                checkpoint.get(key, -1)
            )

            if actual_value != expected_value:
                raise ValueError(
                    "Checkpoint metadata does not match "
                    f"the current model: key={key!r}, "
                    f"checkpoint={actual_value}, "
                    f"expected={expected_value}, "
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

    @staticmethod
    def _to_numpy(
        value: Any,
    ) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return (
                value
                .detach()
                .cpu()
                .numpy()
            )

        return np.asarray(value)