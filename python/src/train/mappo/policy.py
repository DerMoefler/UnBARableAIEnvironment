
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

    def _mask_logits(self, logits, mask):
        if mask is not None:
            mask = self._tensor(mask, torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, -torch.inf)
        return logits

    def _action_dist(self, obs, available_actions=None):
        logits = self._mask_logits(
            self.actor(obs), available_actions
        )
        return Categorical(logits=logits)

    def _target_dist(self, obs, target_features, target_mask=None):
        if target_features is None:
            raise ValueError("Attack requires target_features.")

        features = self._tensor(target_features)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.size(0) == 1 and obs.size(0) > 1:
            features = features.expand(obs.size(0), -1, -1)

        logits = self.target_scorer(obs, features)
        logits = self._mask_logits(logits, target_mask)
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
        action_dist = self._action_dist(obs, available_actions)
        action_type = action_dist.sample()
        log_prob = action_dist.log_prob(action_type)

        # Column 1 stores a candidate index, not a BAR unit ID.
        target_index = torch.zeros_like(action_type)
        attack = action_type == self.attack_action_index

        if attack.any():
            if candidate_target_ids is None:
                raise ValueError("Attack requires candidate_target_ids.")

            target_dist = self._target_dist(
                obs[attack],
                self._tensor(target_features)[
                    attack
                ] if self._tensor(target_features).ndim == 3
                else self._tensor(target_features).unsqueeze(0),
                self._tensor(target_mask, torch.bool)[attack]
                if target_mask is not None
                and self._tensor(target_mask).ndim == 2
                else target_mask,
            )

            selected = target_dist.sample()
            target_index[attack] = selected
            log_prob[attack] += target_dist.log_prob(selected)

        actions = torch.stack([action_type, target_index], dim=-1)
        return actions.cpu().numpy(), log_prob.cpu().numpy()

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
        action_dist = self._action_dist(obs, available_actions_batch)
       
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
