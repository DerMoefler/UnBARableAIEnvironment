import numpy as np
import torch
import torch.nn as nn

try:
    import bar_ai
except Exception:  # pragma: no cover - optional binding in lightweight test environments
    bar_ai = None


class _FallbackAction:
    """Compatibility Action object for environments where the pybind module is unavailable."""

    def __init__(self):
        self.unit_id = 0
        self.team_id = 0
        self.ally_team_id = 0
        self.action_id = 0
        self.target_unit_id = 0

    def __getitem__(self, key):
        if key == "action":
            return {
                1: "move_right",
                2: "move_left",
                3: "move_up",
                4: "move_down",
                5: "attack",
            }.get(int(self.action_id), "attack")
        if key == "target_id":
            return int(self.target_unit_id)
        if key == "unit_id":
            return int(self.unit_id)
        if key == "team_id":
            return int(self.team_id)
        if key == "ally_team_id":
            return int(self.ally_team_id)
        raise KeyError(key)


if bar_ai is not None and not hasattr(bar_ai, "Action"):
    bar_ai.Action = _FallbackAction


def check(x):
    """
    Converts input data into a torch tensor if necessary.

    This function acts as a local replacement for
    `onpolicy.algorithms.utils.util.check`.

    Parameters
    ----------
    x : Any
        Input value to be converted. Can be a numpy array, a torch tensor,
        or any object supported by `torch.as_tensor`.

    Returns
    -------
    tensor : torch.Tensor
        The input converted to a torch tensor if needed.

    Examples
    --------
    >>> check(np.array([1.0, 2.0]))
    tensor([1., 2.], dtype=torch.float64)
    >>> check(torch.tensor([1.0]))
    tensor([1.])
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)


def huber_loss(error, delta):
    """
    Computes the Huber loss for a given prediction error.

    This function acts as a local replacement for
    `onpolicy.utils.util.huber_loss`.

    Parameters
    ----------
    error : torch.Tensor
        Prediction error tensor.
    delta : float
        Threshold at which the loss transitions from quadratic to linear.

    Returns
    -------
    loss : torch.Tensor
        Element-wise Huber loss tensor.

    Examples
    --------
    >>> err = torch.tensor([0.5, 2.0])
    >>> huber_loss(err, 1.0)
    tensor([0.1250, 1.5000])
    """
    abs_error = torch.abs(error)
    quadratic = torch.minimum(
        abs_error,
        torch.tensor(delta, device=error.device, dtype=error.dtype)
    )
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def mse_loss(error):
    """
    Computes the element-wise mean squared error term.

    This function acts as a local replacement for
    `onpolicy.utils.util.mse_loss`.

    Parameters
    ----------
    error : torch.Tensor
        Prediction error tensor.

    Returns
    -------
    loss : torch.Tensor
        Element-wise squared error tensor.

    Examples
    --------
    >>> mse_loss(torch.tensor([1.0, -2.0]))
    tensor([1., 4.])
    """
    return error ** 2


def get_grad_norm(parameters):
    """
    Computes the global L2 norm of gradients for a parameter collection.

    This function acts as a local replacement for
    `onpolicy.utils.util.get_gard_norm`.

    Parameters
    ----------
    parameters : iterable
        Iterable containing model parameters.

    Returns
    -------
    grad_norm : torch.Tensor
        Scalar tensor containing the total gradient norm. Returns 0 if
        no parameter has a gradient.

    Examples
    --------
    >>> model = nn.Linear(2, 1)
    >>> x = torch.tensor([[1.0, 2.0]])
    >>> y = model(x).sum()
    >>> y.backward()
    >>> norm = get_grad_norm(model.parameters())
    >>> isinstance(norm, torch.Tensor)
    True
    """
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)

    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5
    return torch.tensor(total_norm)


class ValueNorm(nn.Module):
    """
    Minimal local replacement for `onpolicy.utils.valuenorm.ValueNorm`.

    This module tracks running mean and variance statistics and can be used
    to normalize or denormalize value targets. In the current setup it is
    included mainly for completeness, because value normalization is disabled
    by default.

    Parameters
    ----------
    input_shape : int or tuple
        Shape of the value tensor to normalize.
    device : torch.device, optional
        Torch device on which internal statistics are stored,
        by default torch.device("cpu").
    epsilon : float, optional
        Small positive constant for numerical stability,
        by default 1e-5.

    Returns
    -------
    ValueNorm
        A normalization module with running statistics.

    Examples
    --------
    >>> vn = ValueNorm(1)
    >>> x = torch.tensor([[1.0], [2.0], [3.0]])
    >>> vn.update(x)
    >>> y = vn.normalize(x)
    >>> y.shape
    torch.Size([3, 1])
    """

    def __init__(self, input_shape, device=torch.device("cpu"), epsilon=1e-5):
        """
        Initializes the value normalization module.

        Parameters
        ----------
        self : ValueNorm
            The normalization module instance.
        input_shape : int or tuple
            Shape of the values to normalize.
        device : torch.device, optional
            Torch device on which internal tensors are allocated,
            by default torch.device("cpu").
        epsilon : float, optional
            Small positive constant for numerical stability,
            by default 1e-5.

        Returns
        -------
        None

        Examples
        --------
        >>> vn = ValueNorm(1)
        >>> vn.epsilon
        1e-05
        """
        super().__init__()
        self.input_shape = input_shape
        self.device = device
        self.epsilon = epsilon

        self.running_mean = torch.zeros(input_shape, device=device, dtype=torch.float32)
        self.running_var = torch.ones(input_shape, device=device, dtype=torch.float32)
        self.count = torch.tensor(epsilon, device=device, dtype=torch.float32)

    def update(self, x):
        """
        Updates running mean and variance statistics from a new batch.

        Parameters
        ----------
        self : ValueNorm
            The normalization module instance.
        x : np.ndarray or torch.Tensor
            Batch of values used to update the running statistics.

        Returns
        -------
        None

        Examples
        --------
        >>> vn = ValueNorm(1)
        >>> vn.update(torch.tensor([[1.0], [2.0], [3.0]]))
        >>> vn.count.item() > 0
        True
        """
        x = check(x).to(device=self.device, dtype=torch.float32)
        if x.ndim == 1:
            batch_mean = x.mean()
            batch_var = x.var(unbiased=False)
            batch_count = torch.tensor(float(x.shape[0]), device=self.device)
        else:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_count = torch.tensor(float(x.shape[0]), device=self.device)

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Updates running statistics from batch moments.

        Parameters
        ----------
        self : ValueNorm
            The normalization module instance.
        batch_mean : torch.Tensor
            Mean of the current batch.
        batch_var : torch.Tensor
            Variance of the current batch.
        batch_count : torch.Tensor
            Number of elements in the batch.

        Returns
        -------
        None

        Examples
        --------
        >>> vn = ValueNorm(1)
        >>> vn._update_from_moments(torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor(3.0))
        >>> vn.running_mean.shape
        torch.Size([1])
        """
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        new_mean = self.running_mean + delta * batch_count / total_count

        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.running_mean = new_mean
        self.running_var = new_var
        self.count = total_count

    def normalize(self, x):
        """
        Normalizes input values using running statistics.

        Parameters
        ----------
        self : ValueNorm
            The normalization module instance.
        x : np.ndarray or torch.Tensor
            Input values to normalize.

        Returns
        -------
        normalized_x : torch.Tensor
            Normalized tensor.

        Examples
        --------
        >>> vn = ValueNorm(1)
        >>> vn.update(torch.tensor([[1.0], [2.0], [3.0]]))
        >>> vn.normalize(torch.tensor([[2.0]])).shape
        torch.Size([1, 1])
        """
        x = check(x).to(device=self.device, dtype=torch.float32)
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

    def denormalize(self, x):
        """
        Converts normalized values back to the original scale.

        Parameters
        ----------
        self : ValueNorm
            The normalization module instance.
        x : np.ndarray or torch.Tensor
            Normalized values.

        Returns
        -------
        denormalized_x : torch.Tensor
            Denormalized tensor in the original value space.

        Examples
        --------
        >>> vn = ValueNorm(1)
        >>> vn.update(torch.tensor([[1.0], [2.0], [3.0]]))
        >>> z = vn.normalize(torch.tensor([[2.0]]))
        >>> vn.denormalize(z).shape
        torch.Size([1, 1])
        """
        x = check(x).to(device=self.device, dtype=torch.float32)
        return x * torch.sqrt(self.running_var + self.epsilon) + self.running_mean


class R_MAPPO:
    """
    Trainer class for MAPPO policy updates.

    This class performs PPO-style actor-critic optimization for multi-agent
    policies, including optional value clipping, Huber loss, recurrent data
    handling, and value normalization.

    Parameters
    ----------
    args : Any
        Configuration object containing optimizer and training hyperparameters.
    policy : Any
        Policy object to be optimized. Must provide actor, critic, optimizers,
        and an `evaluate_actions` method.
    device : torch.device, optional
        Torch device on which training is performed, by default torch.device("cpu").

    Returns
    -------
    R_MAPPO
        A trainer instance for MAPPO optimization.

    Examples
    --------
    >>> # trainer = R_MAPPO(args, policy)
    >>> # trainer.prep_training()
    """

    def __init__(self, args, policy, device=torch.device("cpu")):
        """
        Initializes the MAPPO trainer.

        Parameters
        ----------
        self : R_MAPPO
            The trainer instance.
        args : Any
            Configuration object containing PPO and optimization parameters.
        policy : Any
            Policy object to update.
        device : torch.device, optional
            Torch device used for training, by default torch.device("cpu").

        Returns
        -------
        None

        Examples
        --------
        >>> # trainer = R_MAPPO(args, policy, device=torch.device("cpu"))
        """
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
            "self._use_popart and self._use_valuenorm cannot both be True at the same time"
        )

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def decode_action(self, action_id, enemy_id=None):
        """
        Decodes a discrete action ID into a human-readable command.

        Action mapping:
        - 0: move north
        - 1: move south
        - 2: move east
        - 3: move west
        - 4: attack the selected enemy

        Parameters
        ----------
        self : R_MAPPO
            The trainer instance.
        action_id : int or torch.Tensor
            The action ID to decode.
        enemy_id : int, optional
            Enemy ID for attack actions. If None, derived from action_id.

        Returns
        -------
        action_command : dict
            Dictionary with key:
            - "action": str - Action command (move_north, move_south, move_east, move_west, or attack)
            - "target_id": int or None - Target enemy ID for attack actions

        Examples
        --------
        >>> trainer.decode_action(0)
        {'action': 'move_north'}
        >>> trainer.decode_action(4, enemy_id=5)
        {'action': 'attack', 'target_id': 5}
        """
        if torch.is_tensor(action_id):
            action_id = action_id.item()
        action_id = int(action_id)

        if action_id == 0:
            return {"action": "move_north"}
        elif action_id == 1:
            return {"action": "move_south"}
        elif action_id == 2:
            return {"action": "move_east"}
        elif action_id == 3:
            return {"action": "move_west"}
        else:  # action_id == 4 (attack)
            if enemy_id is None:
                enemy_id = 0
            return {"action": "attack", "target_id": int(enemy_id)}

    def decode_action_to_engine_action(self, action_id, unit_id=0, team_id=0, ally_team_id=0, target_unit_id=0):
        """Convert a PPO action to the engine Action struct used by the C++ side."""
        if torch.is_tensor(action_id):
            action_id = action_id.item()
        action_id = int(action_id)

        if bar_ai is None:
            return {
                "unit_id": int(unit_id),
                "team_id": int(team_id),
                "ally_team_id": int(ally_team_id),
                "action_id": self._map_policy_action_to_engine_action(action_id),
                "target_unit_id": int(target_unit_id),
            }

        action = bar_ai.Action()
        action.unit_id = int(unit_id)
        action.team_id = int(team_id)
        action.ally_team_id = int(ally_team_id)
        action.action_id = self._map_policy_action_to_engine_action(action_id)
        action.target_unit_id = int(target_unit_id)
        return action

    @staticmethod
    def _map_policy_action_to_engine_action(action_id):
        """Map the PPO cardinal-direction action IDs to the engine action contract."""
        mapping = {
            0: 3,  # north -> move up
            1: 4,  # south -> move down
            2: 1,  # east -> move right
            3: 2,  # west -> move left
            4: 5,  # attack
        }
        return int(mapping.get(int(action_id), 5))

    def decode_actions_batch(self, actions_batch, unit_ids=None, team_ids=None, ally_team_ids=None):
        """
        Decodes a batch of PPO actions into engine Action objects.

        The returned values match the C++ struct contract in the engine and can be
        sent directly to the environment.
        """
        if torch.is_tensor(actions_batch):
            actions_batch = actions_batch.detach().cpu().numpy()

        actions_batch = np.asarray(actions_batch)
        if actions_batch.ndim == 1:
            actions_batch = actions_batch.reshape(-1, 1)

        if unit_ids is None:
            unit_ids = np.arange(actions_batch.shape[0], dtype=np.int64)
        if team_ids is None:
            team_ids = np.zeros(actions_batch.shape[0], dtype=np.int64)
        if ally_team_ids is None:
            ally_team_ids = np.zeros(actions_batch.shape[0], dtype=np.int64)

        decoded = []
        for i, action in enumerate(actions_batch):
            action_id = int(action[0])
            target_unit_id = int(action[1]) if action.shape[0] > 1 else 0
            decoded.append(
                self.decode_action_to_engine_action(
                    action_id=action_id,
                    unit_id=int(unit_ids[i]),
                    team_id=int(team_ids[i]),
                    ally_team_id=int(ally_team_ids[i]),
                    target_unit_id=target_unit_id,
                )
            )
        return decoded

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculates the critic value loss.

        Supports optional value clipping, Huber loss, value normalization, and
        masking of inactive agents.

        Parameters
        ----------
        self : R_MAPPO
            The trainer instance.
        values : torch.Tensor
            Current critic predictions.
        value_preds_batch : torch.Tensor
            Value predictions stored in the replay buffer.
        return_batch : torch.Tensor
            Computed return targets.
        active_masks_batch : torch.Tensor
            Masks indicating which agents are active.

        Returns
        -------
        value_loss : torch.Tensor
            Scalar tensor representing the critic loss.

        Examples
        --------
        >>> # loss = trainer.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
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
            denom = active_masks_batch.sum().clamp(min=1e-8)
            value_loss = (value_loss * active_masks_batch).sum() / denom
        else:
            value_loss = value_loss.mean()

        return value_loss

    def _flatten_first_two_dims(self, x):
        """
        Flattens the first two tensor dimensions if possible.

        Converts an input with shape `(batch, num_agents, ...)` into
        `(batch * num_agents, ...)`. Inputs with fewer than three dimensions
        remain unchanged.

        Parameters
        ----------
        self : R_MAPPO
            The trainer instance.
        x : np.ndarray or torch.Tensor or Any
            Input array or tensor to flatten.

        Returns
        -------
        flattened_x : torch.Tensor or None
            Flattened torch tensor, or None if the input was None.

        Examples
        --------
        >>> # x.shape = (4, 2, 8) -> result.shape = (8, 8)
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
    ):
        """
        Prepares and reshapes batch inputs for multi-agent PPO updates.

        This method converts all inputs to torch tensors and ensures they have
        compatible shapes for policy evaluation and loss computation. It supports
        both single-agent and multi-agent batch layouts.

        Parameters
        ----------
        self : R_MAPPO
            The trainer instance.
        share_obs_batch : np.ndarray or torch.Tensor
            Shared observations for the critic.
        obs_batch : np.ndarray or torch.Tensor
            Agent-specific observations.
        rnn_states_batch : np.ndarray or torch.Tensor or None
            Actor recurrent states.
        rnn_states_critic_batch : np.ndarray or torch.Tensor or None
            Critic recurrent states.
        actions_batch : np.ndarray or torch.Tensor
            Actions from the replay buffer.
        value_preds_batch : np.ndarray or torch.Tensor
            Stored critic predictions.
        return_batch : np.ndarray or torch.Tensor
            Computed return targets.
        masks_batch : np.ndarray or torch.Tensor
            Episode continuation masks.
        active_masks_batch : np.ndarray or torch.Tensor
            Agent activity masks.
        old_action_log_probs_batch : np.ndarray or torch.Tensor
            Log-probabilities from the behavior policy.
        adv_targ : np.ndarray or torch.Tensor
            Advantage targets.
        available_actions_batch : np.ndarray or torch.Tensor or None
            Legal action masks.

        Returns
        -------
        prepared_batches : tuple
            Tuple containing all prepared tensors in the correct shape.

        Examples
        --------
        >>> # prepared = trainer._prepare_multi_agent_inputs(...)
        >>> # len(prepared)
        >>> # 12
        """
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

        if rnn_states_batch is not None:
            rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        if rnn_states_critic_batch is not None:
            rnn_states_critic_batch = check(rnn_states_critic_batch).to(**self.tpdv)

        if obs_batch.ndim == 3:
            batch_size, num_agents = obs_batch.shape[0], obs_batch.shape[1]

            if share_obs_batch.ndim == 2:
                if share_obs_batch.shape[0] != batch_size:
                    raise ValueError(
                        f"Expected share_obs_batch.shape[0] == batch_size, but got "
                        f"{share_obs_batch.shape} vs obs_batch {obs_batch.shape}"
                    )
                share_obs_batch = share_obs_batch.unsqueeze(1).expand(-1, num_agents, -1)

            elif share_obs_batch.ndim == 3:
                if share_obs_batch.shape[0] != batch_size or share_obs_batch.shape[1] != num_agents:
                    raise ValueError(
                        f"share_obs_batch shape {share_obs_batch.shape} is incompatible "
                        f"with obs_batch shape {obs_batch.shape}"
                    )
            else:
                raise ValueError(
                    f"Unsupported share_obs_batch.ndim={share_obs_batch.ndim} for multi-agent batch"
                )

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
        Performs a single PPO update step for actor and critic.

        Parameters
        ----------
        self : R_MAPPO
            The trainer instance.
        sample : tuple
            A mini-batch tuple produced by the replay buffer generator.
        update_actor : bool, optional
            Whether to backpropagate the actor loss, by default True.

        Returns
        -------
        update_results : tuple
            Tuple containing:
            - value_loss : torch.Tensor
            - critic_grad_norm : torch.Tensor or float
            - policy_loss : torch.Tensor
            - dist_entropy : torch.Tensor
            - actor_grad_norm : torch.Tensor or float
            - imp_weights : torch.Tensor
            - decoded_actions : list of dict - Decoded action commands for engine

        Examples
        --------
        >>> # results = trainer.ppo_update(sample)
        >>> # len(results) == 7
        >>> # results[6][0]  # {'action': 'move_north'} or {'action': 'attack', 'target_id': 5}
        """
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
        )

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

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(
            imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param
        ) * adv_targ

        if self._use_policy_active_masks:
            denom = active_masks_batch.sum().clamp(min=1e-8)
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / denom
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        actor_parameters = list(self.policy.actor.parameters())
        if hasattr(self.policy, "target_actor"):
            actor_parameters.extend(self.policy.target_actor.parameters())

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                actor_parameters, self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(actor_parameters)

        self.policy.actor_optimizer.step()

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
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        # Decode actions for engine output
        decoded_actions = self.decode_actions_batch(
            actions_batch,
            unit_ids=np.arange(actions_batch.shape[0], dtype=np.int64),
            team_ids=np.zeros(actions_batch.shape[0], dtype=np.int64),
            ally_team_ids=np.zeros(actions_batch.shape[0], dtype=np.int64),
        )

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
            decoded_actions,
        )

    def train(self, buffer, update_actor=True):
        """
        Performs one full MAPPO training phase using mini-batch gradient descent.

        The method computes normalized advantages, iterates over PPO epochs,
        generates mini-batches from the replay buffer, and aggregates training
        statistics with decoded actions for the game engine.

        Parameters
        ----------
        self : R_MAPPO
            The trainer instance.
        buffer : SharedReplayBuffer
            Replay buffer containing rollout data.
        update_actor : bool, optional
            Whether to update the actor network, by default True.

        Returns
        -------
        train_info : dict
            Dictionary containing averaged training statistics:
            - `value_loss`: Average critic loss
            - `policy_loss`: Average actor loss
            - `dist_entropy`: Average policy entropy
            - `actor_grad_norm`: Average actor gradient norm
            - `critic_grad_norm`: Average critic gradient norm
            - `ratio`: Average importance weight ratio
            - `actions`: List of all decoded action commands (dicts with 'action' and optional 'target_id')

        Examples
        --------
        >>> train_info = trainer.train(buffer)
        >>> train_info["value_loss"]  # 0.125
        >>> train_info["actions"][0]  # {'action': 'move_north'}
        >>> train_info["actions"][1]  # {'action': 'attack', 'target_id': 3}
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

        train_info = {
            "value_loss": 0,
            "policy_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "critic_grad_norm": 0,
            "ratio": 0,
        }

        all_actions = []

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
                    decoded_actions,
                ) = self.ppo_update(sample, update_actor)

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += float(actor_grad_norm)
                train_info["critic_grad_norm"] += float(critic_grad_norm)
                train_info["ratio"] += imp_weights.mean().item()
                all_actions.extend(decoded_actions)

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        train_info["actions"] = all_actions

        return train_info

    def prep_training(self):
        """
        Switches actor and critic networks to training mode.

        Parameters
        ----------
        self : R_MAPPO
            The trainer instance.

        Returns
        -------
        None

        Examples
        --------
        >>> # trainer.prep_training()
        """
        self.policy.actor.train()
        if hasattr(self.policy, "target_actor"):
            self.policy.target_actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """
        Switches actor and critic networks to evaluation mode.

        Parameters
        ----------
        self : R_MAPPO
            The trainer instance.

        Returns
        -------
        None

        Examples
        --------
        >>> # trainer.prep_rollout()
        """
        self.policy.actor.eval()
        if hasattr(self.policy, "target_actor"):
            self.policy.target_actor.eval()
        self.policy.critic.eval()