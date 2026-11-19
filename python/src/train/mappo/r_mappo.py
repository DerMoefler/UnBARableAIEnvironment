import numpy as np
import torch
import torch.nn as nn
from .utils import ValueNorm, check, get_grad_norm, huber_loss, mse_loss


class R_MAPPO:
    """Trainer class for MAPPO to update policies."""

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

        assert not (self._use_popart and self._use_valuenorm), (
            "self._use_popart and self._use_valuenorm cannot both be set True"
        )

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            active_agent_count = active_masks_batch.sum()

            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_agent_count.clamp_min(1.0)

        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        if len(sample) == 14:
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
                candidate_target_features_batch,
                candidate_target_mask_batch,
            ) = sample

        elif len(sample) == 13:
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
                candidate_target_features_batch,
            ) = sample
            candidate_target_mask_batch = None

        elif len(sample) == 12:
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
            candidate_target_features_batch = None
            candidate_target_mask_batch = None

        else:
            raise ValueError(
                f"Unexpected PPO sample length: {len(sample)}"
            )

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            candidate_target_features_batch=candidate_target_features_batch,
            candidate_target_mask_batch=candidate_target_mask_batch,
        )

        value_active_masks_batch = active_masks_batch
        if return_batch.ndim == values.ndim + 1:
            agent_masks = active_masks_batch
            active_agent_count = agent_masks.sum(dim=-2)
            value_preds_batch = value_preds_batch.mean(dim=-2)
            return_batch = (return_batch * agent_masks).sum(dim=-2) / (
                active_agent_count.clamp_min(1.0)
            )
            value_active_masks_batch = (active_agent_count > 0).to(
                dtype=values.dtype
            )

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(
            imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param
        ) * adv_targ

        if self._use_policy_active_masks:
            active_agent_count = active_masks_batch.sum()

            policy_action_loss = (
                -torch.sum(
                    torch.min(surr1, surr2),
                    dim=-1,
                    keepdim=True,
                )
                * active_masks_batch
            ).sum() / active_agent_count.clamp_min(1.0)

        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2),
                dim=-1,
                keepdim=True,
            ).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        has_active_agents = bool(
            torch.any(active_masks_batch > 0).item()
        )

        if update_actor and has_active_agents:
            (
                policy_loss
                - dist_entropy * self.entropy_coef
            ).backward()

        actor_parameters = (
            list(self.policy.actor.parameters())
            + list(self.policy.target_scorer.parameters())
        )

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                actor_parameters,
                self.max_grad_norm,
            )
        else:
            actor_grad_norm = get_grad_norm(
                actor_parameters
            )

        if update_actor and has_active_agents:
            self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, value_active_masks_batch
        )
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):

        log_callback = getattr(self, "log_callback", None)
        log_level = getattr(self, "log_level", 1)
        
        valid_steps = buffer.step

        if valid_steps <= 0:
            raise RuntimeError(
                "Cannot train MAPPO with an empty replay buffer."
            )

        if self._use_popart or self._use_valuenorm:
            advantages = (
                buffer.returns[:valid_steps]
                - self.value_normalizer.denormalize(
                    buffer.value_preds[:valid_steps]
                )
            )
        else:
            advantages = (
                buffer.returns[:valid_steps]
                - buffer.value_preds[:valid_steps]
            )

        advantages_copy = advantages.copy()

        active_masks = buffer.active_masks[:valid_steps]

        advantages_copy[active_masks == 0.0] = np.nan

        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)

        if not np.isfinite(mean_advantages):
            raise FloatingPointError(
                "Advantage mean is not finite. "
                "All samples may be inactive."
            )

        if not np.isfinite(std_advantages):
            raise FloatingPointError(
                "Advantage standard deviation is not finite."
            )

        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {
            "value_loss": 0,
            "policy_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "critic_grad_norm": 0,
            "ratio": 0,
        }

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
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for key in train_info:
            train_info[key] /= num_updates

        if log_callback is not None and log_level >= 2:
            log_callback(
                2,
                "ppo_update_completed",
                **{
                    key: float(value)
                    for key, value in train_info.items()
                },
            )

        return train_info

    def train_multiple(
        self,
        buffers,
        update_actor=True,
    ):
        """
        Train one shared policy with rollout data from multiple teams.

        Returns and GAE are expected to have been computed separately for
        every buffer before this method is called.

        Advantages are normalized jointly across all active agent samples.
        PPO samples from every team are then combined and globally shuffled.
        """
        buffers = list(buffers)

        if not buffers:
            raise ValueError(
                "train_multiple requires at least one "
                "replay buffer."
            )

        for buffer_index, buffer in enumerate(
            buffers
        ):
            if buffer.step <= 0:
                raise RuntimeError(
                    "Cannot train MAPPO with an empty "
                    "replay buffer: "
                    f"buffer_index={buffer_index}."
                )

        log_callback = getattr(
            self,
            "log_callback",
            None,
        )

        log_level = getattr(
            self,
            "log_level",
            1,
        )

        # ---------------------------------------------------------
        # Calculate advantages separately per trajectory
        # ---------------------------------------------------------

        raw_advantages = [
            self._calculate_buffer_advantages(
                buffer
            )
            for buffer in buffers
        ]

        # ---------------------------------------------------------
        # Calculate one shared normalization over active samples
        # ---------------------------------------------------------

        active_advantage_values = []

        for buffer, advantages in zip(
            buffers,
            raw_advantages,
        ):
            valid_steps = buffer.step

            active_masks = np.asarray(
                buffer.active_masks[
                    :valid_steps
                ],
                dtype=np.float32,
            )

            if active_masks.shape != (
                advantages.shape
            ):
                raise ValueError(
                    "Active-mask shape does not match "
                    "advantage shape: "
                    f"active_masks={active_masks.shape}, "
                    f"advantages={advantages.shape}."
                )

            active_values = advantages[
                active_masks > 0.0
            ]

            if active_values.size > 0:
                active_advantage_values.append(
                    active_values
                )

        if not active_advantage_values:
            raise FloatingPointError(
                "No active agent samples are available "
                "for advantage normalization."
            )

        combined_active_advantages = (
            np.concatenate(
                active_advantage_values,
                axis=0,
            )
        )

        mean_advantages = float(
            np.mean(
                combined_active_advantages
            )
        )

        std_advantages = float(
            np.std(
                combined_active_advantages
            )
        )

        if not np.isfinite(
            mean_advantages
        ):
            raise FloatingPointError(
                "Combined advantage mean is not finite."
            )

        if not np.isfinite(
            std_advantages
        ):
            raise FloatingPointError(
                "Combined advantage standard deviation "
                "is not finite."
            )

        normalized_advantages = [
            (
                advantages
                - mean_advantages
            ) / (
                std_advantages
                + 1e-5
            )
            for advantages in raw_advantages
        ]

        # ---------------------------------------------------------
        # PPO optimization
        # ---------------------------------------------------------

        train_info = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dist_entropy": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "ratio": 0.0,
        }

        num_updates = 0

        for _ in range(
            self.ppo_epoch
        ):
            data_generator = (
                self._combined_feed_forward_generator(
                    buffers,
                    normalized_advantages,
                )
            )

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(
                    sample,
                    update_actor,
                )

                train_info[
                    "value_loss"
                ] += float(
                    value_loss.item()
                )

                train_info[
                    "policy_loss"
                ] += float(
                    policy_loss.item()
                )

                train_info[
                    "dist_entropy"
                ] += float(
                    dist_entropy.item()
                )

                train_info[
                    "actor_grad_norm"
                ] += float(
                    actor_grad_norm
                )

                train_info[
                    "critic_grad_norm"
                ] += float(
                    critic_grad_norm
                )

                train_info[
                    "ratio"
                ] += float(
                    imp_weights.mean().item()
                )

                num_updates += 1

        if num_updates <= 0:
            raise RuntimeError(
                "Combined PPO training produced no updates."
            )

        for key in train_info:
            train_info[key] /= num_updates

        if (
            log_callback is not None
            and log_level >= 2
        ):
            log_callback(
                2,
                "ppo_update_completed",
                **train_info,
            )

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

    def _calculate_buffer_advantages(
        self,
        buffer,
    ) -> np.ndarray:
        """
        Calculate unnormalized advantages for one rollout buffer.
        """
        valid_steps = buffer.step

        if valid_steps <= 0:
            raise RuntimeError(
                "Cannot calculate advantages for an empty "
                "replay buffer."
            )

        if (
            self._use_popart
            or self._use_valuenorm
        ):
            advantages = (
                buffer.returns[:valid_steps]
                - self.value_normalizer.denormalize(
                    buffer.value_preds[:valid_steps]
                )
            )
        else:
            advantages = (
                buffer.returns[:valid_steps]
                - buffer.value_preds[:valid_steps]
            )

        return np.asarray(
            advantages,
            dtype=np.float32,
        )

    @staticmethod
    def _concatenate_samples(
        samples: list[tuple],
    ) -> tuple:
        """
        Concatenate complete flattened PPO samples from multiple buffers.

        Every non-None field is concatenated along the sample dimension.
        Optional target fields must either be present in every sample or
        absent from every sample.
        """
        if not samples:
            raise ValueError(
                "Cannot concatenate an empty sample list."
            )

        sample_length = len(samples[0])

        if any(
            len(sample) != sample_length
            for sample in samples
        ):
            raise ValueError(
                "All PPO samples must contain the same "
                "number of fields."
            )

        combined_fields = []

        for field_index in range(sample_length):
            field_values = [
                sample[field_index]
                for sample in samples
            ]

            none_count = sum(
                value is None
                for value in field_values
            )

            if none_count == len(field_values):
                combined_fields.append(None)
                continue

            if none_count != 0:
                raise ValueError(
                    "Inconsistent optional PPO sample field: "
                    f"field_index={field_index}, "
                    "some buffers contain data and others "
                    "contain None."
                )

            arrays = [
                np.asarray(value)
                for value in field_values
            ]

            reference_shape = arrays[0].shape[1:]

            for array in arrays[1:]:
                if array.shape[1:] != reference_shape:
                    raise ValueError(
                        "Cannot combine PPO sample fields "
                        "with different non-batch dimensions: "
                        f"field_index={field_index}, "
                        f"expected={reference_shape}, "
                        f"actual={array.shape[1:]}."
                    )

            combined_fields.append(
                np.concatenate(
                    arrays,
                    axis=0,
                )
            )

        return tuple(combined_fields)

    def _combined_feed_forward_generator(
        self,
        buffers,
        advantages_by_buffer,
    ):
        """
        Flatten all team buffers, combine their samples and generate
        globally shuffled PPO mini-batches.
        """
        complete_samples = []

        for buffer, advantages in zip(
            buffers,
            advantages_by_buffer,
        ):
            # Ein Mini-Batch liefert hier den vollständigen
            # abgeflachten Inhalt dieses Buffers.
            generator = buffer.feed_forward_generator(
                advantages,
                num_mini_batch=1,
            )

            try:
                complete_sample = next(
                    generator
                )
            except StopIteration as exc:
                raise RuntimeError(
                    "Replay buffer produced no PPO samples."
                ) from exc

            complete_samples.append(
                complete_sample
            )

        combined_sample = (
            self._concatenate_samples(
                complete_samples
            )
        )

        first_field = combined_sample[0]

        if first_field is None:
            raise RuntimeError(
                "Combined PPO sample has no observations."
            )

        total_samples = int(
            np.asarray(first_field).shape[0]
        )

        if total_samples <= 0:
            raise RuntimeError(
                "Combined PPO sample is empty."
            )

        if self.num_mini_batch > total_samples:
            raise ValueError(
                "num_mini_batch cannot exceed the total "
                "number of combined PPO samples: "
                f"num_mini_batch={self.num_mini_batch}, "
                f"total_samples={total_samples}."
            )

        permutation = np.random.permutation(
            total_samples
        )

        mini_batches = np.array_split(
            permutation,
            self.num_mini_batch,
        )

        for batch_indices in mini_batches:
            if batch_indices.size == 0:
                continue

            mini_batch = tuple(
                None
                if field is None
                else np.asarray(field)[
                    batch_indices
                ]
                for field in combined_sample
            )

            yield mini_batch
            