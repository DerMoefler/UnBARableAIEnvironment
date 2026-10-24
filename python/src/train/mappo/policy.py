
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

MOVEMENT_TARGET_ID = 4211


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.v_out = self.net

    def forward(self, obs):
        return self.net(obs)


class TargetScorer(nn.Module):
    """Scores a variable number of target candidates."""

    def __init__(self, obs_dim, target_feature_dim, hidden_dim=128):
        super().__init__()
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU()
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(target_feature_dim, hidden_dim), nn.ReLU()
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, target_features):
        # obs: [B, obs_dim]
        # target_features: [B, N, target_feature_dim]
        agent = self.agent_encoder(obs).unsqueeze(1)
        targets = self.target_encoder(target_features)
        agent = agent.expand(-1, targets.size(1), -1)
        return self.scorer(torch.cat([agent, targets], dim=-1)).squeeze(-1)


class R_MAPPO_Policy:
    def __init__(
        self,
        obs_dim,
        action_dim,
        device=torch.device("cpu"),
        lr=5e-4,
        target_feature_dim=None,
        attack_action_index=None,
        hidden_dim=256,
        target_hidden_dim=128,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.attack_action_index = (
            action_dim - 1 if attack_action_index is None
            else attack_action_index
        )

        target_feature_dim = target_feature_dim or obs_dim

        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_scorer = TargetScorer(
            obs_dim, target_feature_dim, target_hidden_dim
        ).to(self.device)
        self.critic = Critic(obs_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.target_scorer.parameters()),
            lr=lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr
        )

    def _tensor(self, value, dtype=torch.float32):
        if value is None:
            return None
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _batch_obs(self, obs):
        obs = self._tensor(obs)
        return obs.unsqueeze(0) if obs.ndim == 1 else obs

    def _mask_logits(self, logits, mask, *, name="distribution"):
        if not torch.isfinite(logits).all():
            bad_indices = (~torch.isfinite(logits)).nonzero(
                as_tuple=False
            )

            raise FloatingPointError(
                f"{name} produced non-finite logits before masking. "
                f"Bad indices: {bad_indices.detach().cpu().tolist()}, "
                f"logits: {logits.detach().cpu().tolist()}"
            )

        if mask is None:
            return logits

        mask = self._tensor(mask, torch.bool)

        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        if mask.shape != logits.shape:
            raise ValueError(
                f"{name} mask shape {tuple(mask.shape)} does not match "
                f"logits shape {tuple(logits.shape)}."
            )

        valid_per_row = mask.any(dim=-1)

        if not valid_per_row.all():
            invalid_rows = (~valid_per_row).nonzero(
                as_tuple=False
            ).flatten()

            raise ValueError(
                f"{name} has rows without any valid option: "
                f"{invalid_rows.detach().cpu().tolist()}. "
                f"Mask: {mask.detach().cpu().tolist()}"
            )

        return logits.masked_fill(
            ~mask,
            torch.finfo(logits.dtype).min,
        )

    def _effective_action_mask(
        self,
        obs,
        available_actions=None,
        target_mask=None,
    ):
        batch_size = obs.size(0)

        if available_actions is None:
            action_mask = torch.ones(
                (batch_size, self.action_dim),
                dtype=torch.bool,
                device=self.device,
            )
        else:
            action_mask = self._tensor(
                available_actions,
                torch.bool,
            )

            if action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0)

            if action_mask.size(0) == 1 and batch_size > 1:
                action_mask = action_mask.expand(
                    batch_size,
                    -1,
                )

            if action_mask.shape != (
                batch_size,
                self.action_dim,
            ):
                raise ValueError(
                    "available_actions has an invalid shape: "
                    f"expected {(batch_size, self.action_dim)}, "
                    f"got {tuple(action_mask.shape)}."
                )

            action_mask = action_mask.clone()

        if target_mask is not None:
            target_mask_tensor = self._tensor(
                target_mask,
                torch.bool,
            )

            if target_mask_tensor.ndim == 1:
                target_mask_tensor = target_mask_tensor.unsqueeze(0)

            if (
                target_mask_tensor.size(0) == 1
                and batch_size > 1
            ):
                target_mask_tensor = target_mask_tensor.expand(
                    batch_size,
                    -1,
                )

            if target_mask_tensor.size(0) != batch_size:
                raise ValueError(
                    "target_mask batch size does not match observations: "
                    f"obs batch={batch_size}, "
                    f"target mask={tuple(target_mask_tensor.shape)}."
                )

            has_valid_target = target_mask_tensor.any(dim=-1)

            # Angriff für Agents ohne gültiges Ziel deaktivieren.
            action_mask[
                ~has_valid_target,
                self.attack_action_index,
            ] = False

        valid_actions = action_mask.any(dim=-1)

        if not valid_actions.all():
            invalid_rows = (~valid_actions).nonzero(
                as_tuple=False
            ).flatten()

            raise ValueError(
                "Agents have no valid action after target-aware masking: "
                f"rows={invalid_rows.detach().cpu().tolist()}, "
                f"mask={action_mask.detach().cpu().tolist()}."
            )

        return action_mask

    def _action_dist(
        self,
        obs,
        available_actions=None,
        target_mask=None,
    ):
        effective_mask = self._effective_action_mask(
            obs,
            available_actions=available_actions,
            target_mask=target_mask,
        )

        logits = self.actor(obs)
        logits = self._mask_logits(
            logits,
            effective_mask,
            name="action distribution",
        )

        return Categorical(logits=logits)

    def _target_dist(
        self,
        obs,
        target_features,
        target_mask=None,
    ):
        if target_features is None:
            raise ValueError(
                "Attack requires target_features."
            )

        features = self._tensor(target_features)

        if features.ndim == 2:
            features = features.unsqueeze(0)

        if features.size(0) == 1 and obs.size(0) > 1:
            features = features.expand(
                obs.size(0),
                -1,
                -1,
            )

        if features.size(0) != obs.size(0):
            raise ValueError(
                "Target-feature batch does not match observation batch: "
                f"obs={tuple(obs.shape)}, "
                f"features={tuple(features.shape)}."
            )

        if not torch.isfinite(features).all():
            bad_indices = (~torch.isfinite(features)).nonzero(
                as_tuple=False
            )

            raise FloatingPointError(
                "Target features contain NaN or infinity. "
                f"Bad indices: {bad_indices.detach().cpu().tolist()}, "
                f"features: {features.detach().cpu().tolist()}"
            )

        logits = self.target_scorer(
            obs,
            features,
        )

        logits = self._mask_logits(
            logits,
            target_mask,
            name="target distribution",
        )

        return Categorical(logits=logits)


    @torch.no_grad()
    def get_action(
        self,
        obs,
        target_features=None,
        candidate_target_ids=None,
        target_mask=None,
        available_actions=None,
    ):
        obs = self._batch_obs(obs)

        if not torch.isfinite(obs).all():
            bad_indices = (~torch.isfinite(obs)).nonzero(
                as_tuple=False
            )

            raise FloatingPointError(
                "Policy observations contain NaN or infinity. "
                f"Bad indices: {bad_indices.detach().cpu().tolist()}"
            )

        target_features_tensor = (
            self._tensor(target_features)
            if target_features is not None
            else None
        )

        target_mask_tensor = (
            self._tensor(target_mask, torch.bool)
            if target_mask is not None
            else None
        )

        if (
            target_features_tensor is not None
            and target_features_tensor.ndim == 2
        ):
            target_features_tensor = (
                target_features_tensor.unsqueeze(0)
            )

        if (
            target_mask_tensor is not None
            and target_mask_tensor.ndim == 1
        ):
            target_mask_tensor = target_mask_tensor.unsqueeze(0)

        # Die Attack-Action wird hier deaktiviert, wenn kein Ziel
        # vorhanden ist.
        action_dist = self._action_dist(
            obs,
            available_actions=available_actions,
            target_mask=target_mask_tensor,
        )

        action_type = action_dist.sample()
        log_prob = action_dist.log_prob(action_type)

        # Spalte 1 enthält einen Kandidatenindex, keine BAR-Unit-ID.
        target_index = torch.zeros_like(action_type)

        attack = action_type == self.attack_action_index

        if attack.any():
            if candidate_target_ids is None:
                raise ValueError(
                    "Attack requires candidate_target_ids."
                )

            if target_features_tensor is None:
                raise ValueError(
                    "Attack requires target_features."
                )

            if target_mask_tensor is None:
                raise ValueError(
                    "Attack requires target_mask."
                )

            attack_features = target_features_tensor[attack]
            attack_mask = target_mask_tensor[attack]

            # Sollte durch _effective_action_mask eigentlich unmöglich sein.
            if not attack_mask.any(dim=-1).all():
                raise RuntimeError(
                    "Attack was selected for an agent without a valid "
                    "target. This indicates inconsistent action masking."
                )

            target_dist = self._target_dist(
                obs[attack],
                attack_features,
                attack_mask,
            )

            selected = target_dist.sample()

            target_index[attack] = selected
            log_prob[attack] += target_dist.log_prob(selected)

        actions = torch.stack(
            [action_type, target_index],
            dim=-1,
        )

        return (
            actions.cpu().numpy(),
            log_prob.unsqueeze(-1).cpu().numpy(),
        )


    def resolve_target_id(self, action_type, target_index, candidate_target_ids):
        """Convert the sampled candidate index to the actual BAR target ID."""
        if int(action_type) != self.attack_action_index:
            return MOVEMENT_TARGET_ID
        return int(candidate_target_ids[int(target_index)])

    def evaluate_actions(
        self,
        share_obs_batch,
        obs_batch,
        rnn_states_batch,
        rnn_states_critic_batch,
        actions_batch,
        masks_batch,
        available_actions_batch,
        active_masks_batch,
        candidate_target_features_batch=None,
        candidate_target_mask_batch=None,
    ):
        share_obs = self._batch_obs(share_obs_batch)
        obs = self._batch_obs(obs_batch)
        actions = self._tensor(actions_batch, torch.long)

        if actions.ndim == 1:
            action_type = actions
            target_index = torch.zeros_like(action_type)
        else:
            action_type = actions[..., 0]
            target_index = actions[..., 1]

        values = self.critic(share_obs)
        action_dist = self._action_dist(
            obs,
            available_actions=available_actions_batch,
            target_mask=candidate_target_mask_batch,
        )
            
        log_probs = action_dist.log_prob(action_type)
        entropy = action_dist.entropy()

        attack = action_type == self.attack_action_index
        if attack.any():
            if candidate_target_features_batch is None:
                raise ValueError(
                    "PPO needs stored target features for attack actions."
                )

            features = self._tensor(candidate_target_features_batch)
            if features.ndim == 2:
                features = features.unsqueeze(0)

            mask = candidate_target_mask_batch
            if mask is not None:
                mask = self._tensor(mask, torch.bool)
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)

            target_dist = self._target_dist(
                obs[attack],
                features[attack],
                mask[attack] if mask is not None else None,
            )
            log_probs[attack] += target_dist.log_prob(target_index[attack])
            entropy[attack] += target_dist.entropy()

        return values, log_probs.unsqueeze(-1), entropy.mean()
