import numpy as np
import torch
import torch.nn as nn


class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert (self._use_popart and self._use_valuenorm) is False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously"
        )

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            from onpolicy.utils.valuenorm import ValueNorm
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value predictions from data batch.
        :param return_batch: (torch.Tensor) reward-to-go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (
            values - value_preds_batch
        ).clamp(-self.clip_param, self.clip_param)

        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            from onpolicy.utils.util import huber_loss
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            from onpolicy.utils.util import mse_loss
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def _flatten_first_two_dims(self, x):
        """
        Flatten (batch, num_agents, ...) -> (batch * num_agents, ...)
        Leaves tensors with ndim < 3 unchanged.
        """
        if x is None:
            return None

        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        elif not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(**self.tpdv)

        if x.ndim >= 3:
            return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        return x

    def _prepare_multi_agent_inputs(
        self,
        share_obs_batch,
        obs_batch,
        rnn_states_batch,
        rnn_states_critic_batch,
        actions_batch,
        value_preds_batch,
        return_batch,
        masks_batch,
        active_masks_batch,
        old_action_log_probs_batch,
        adv_targ,
        available_actions_batch,
        check
    ):
        """
        Prepare tensors for policy evaluation and loss computation.

        Handles these cases:
        - Single-agent batches
        - Multi-agent batches where obs/actions/etc. have shape (B, A, ...)
        - Multi-agent batches where share_obs is global per timestep with shape (B, D)
          and must be repeated across the agent dimension
        """
        # Convert core tensors
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)

        if available_actions_batch is not None:
            available_actions_batch = check(available_actions_batch).to(**self.tpdv)

        # Convert optional RNN states
        if rnn_states_batch is not None:
            rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        if rnn_states_critic_batch is not None:
            rnn_states_critic_batch = check(rnn_states_critic_batch).to(**self.tpdv)

        # ------------------------------------------------------------------
        # Multi-agent case:
        # obs_batch shape is usually (B, A, obs_dim)
        # share_obs_batch may be:
        #   - (B, A, share_obs_dim)  -> already per-agent
        #   - (B, share_obs_dim)     -> global state per timestep, repeat for A
        # ------------------------------------------------------------------
        if obs_batch.ndim == 3:
            batch_size, num_agents = obs_batch.shape[0], obs_batch.shape[1]

            # Make share_obs_batch per-agent before flattening
            if share_obs_batch.ndim == 2:
                # (B, D) -> (B, A, D)
                if share_obs_batch.shape[0] != batch_size:
                    raise ValueError(
                        f"Expected share_obs_batch.shape[0] == batch_size, but got "
                        f"{share_obs_batch.shape} vs obs_batch {obs_batch.shape}"
                    )
                share_obs_batch = share_obs_batch.unsqueeze(1).expand(-1, num_agents, -1)

            elif share_obs_batch.ndim == 3:
                # Already per-agent
                if share_obs_batch.shape[0] != batch_size or share_obs_batch.shape[1] != num_agents:
                    raise ValueError(
                        f"share_obs_batch shape {share_obs_batch.shape} is incompatible "
                        f"with obs_batch shape {obs_batch.shape}"
                    )
            else:
                raise ValueError(
                    f"Unsupported share_obs_batch.ndim={share_obs_batch.ndim} for multi-agent batch"
                )

            # Flatten all per-agent tensors to (B * A, ...)
            share_obs_batch = share_obs_batch.reshape(batch_size * num_agents, -1)
            obs_batch = obs_batch.reshape(batch_size * num_agents, -1)
            actions_batch = actions_batch.reshape(batch_size * num_agents, -1)

            value_preds_batch = value_preds_batch.reshape(batch_size * num_agents, -1)
            return_batch = return_batch.reshape(batch_size * num_agents, -1)
            active_masks_batch = active_masks_batch.reshape(batch_size * num_agents, -1)
            old_action_log_probs_batch = old_action_log_probs_batch.reshape(batch_size * num_agents, -1)
            adv_targ = adv_targ.reshape(batch_size * num_agents, -1)

            if masks_batch.ndim == 3:
                masks_batch = masks_batch.reshape(batch_size * num_agents, -1)

            if available_actions_batch is not None and available_actions_batch.ndim >= 3:
                available_actions_batch = available_actions_batch.reshape(
                    batch_size * num_agents, *available_actions_batch.shape[2:]
                )

            if rnn_states_batch is not None and rnn_states_batch.ndim >= 3:
                rnn_states_batch = rnn_states_batch.reshape(
                    batch_size * num_agents, *rnn_states_batch.shape[2:]
                )

            if rnn_states_critic_batch is not None and rnn_states_critic_batch.ndim >= 3:
                rnn_states_critic_batch = rnn_states_critic_batch.reshape(
                    batch_size * num_agents, *rnn_states_critic_batch.shape[2:]
                )

        else:
            # Single-agent or already flattened path
            if share_obs_batch.ndim > 2:
                share_obs_batch = share_obs_batch.reshape(share_obs_batch.shape[0], -1)
            if obs_batch.ndim > 2:
                obs_batch = obs_batch.reshape(obs_batch.shape[0], -1)
            if actions_batch.ndim > 2:
                actions_batch = actions_batch.reshape(actions_batch.shape[0], -1)

            if value_preds_batch.ndim > 2:
                value_preds_batch = value_preds_batch.reshape(value_preds_batch.shape[0], -1)
            if return_batch.ndim > 2:
                return_batch = return_batch.reshape(return_batch.shape[0], -1)
            if active_masks_batch.ndim > 2:
                active_masks_batch = active_masks_batch.reshape(active_masks_batch.shape[0], -1)
            if old_action_log_probs_batch.ndim > 2:
                old_action_log_probs_batch = old_action_log_probs_batch.reshape(
                    old_action_log_probs_batch.shape[0], -1
                )
            if adv_targ.ndim > 2:
                adv_targ = adv_targ.reshape(adv_targ.shape[0], -1)

            if masks_batch.ndim > 2:
                masks_batch = masks_batch.reshape(masks_batch.shape[0], -1)
            if available_actions_batch is not None and available_actions_batch.ndim > 2:
                available_actions_batch = available_actions_batch.reshape(
                    available_actions_batch.shape[0], -1
                )
            if rnn_states_batch is not None and rnn_states_batch.ndim > 2:
                rnn_states_batch = rnn_states_batch.reshape(rnn_states_batch.shape[0], -1)
            if rnn_states_critic_batch is not None and rnn_states_critic_batch.ndim > 2:
                rnn_states_critic_batch = rnn_states_critic_batch.reshape(
                    rnn_states_critic_batch.shape[0], -1
                )

        return (
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        )

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.

        :param sample: (Tuple) contains data batch with which to update networks.
        :param update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        :return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        from onpolicy.algorithms.utils.util import check
        from onpolicy.utils.util import get_gard_norm

        if len(sample) == 12:
            (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            ) = sample
        else:
            (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                _,
            ) = sample

        (
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = self._prepare_multi_agent_inputs(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            check,
        )

        # Reshape to do in a single forward pass for all steps/agents
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # Actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(
            imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param
        ) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # Critic update
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )

        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
        )

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update.
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info["value_loss"] = 0
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(sample, update_actor)

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += float(actor_grad_norm)
                train_info["critic_grad_norm"] += float(critic_grad_norm)
                train_info["ratio"] += imp_weights.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
