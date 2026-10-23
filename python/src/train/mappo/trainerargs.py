class TrainerArgs:
    """
    Default configuration values for the MAPPO training loop.

    Returns
    -------
    TrainerArgs
        Configuration object containing PPO and rollout settings.

    Examples
    --------
    >>> args = TrainerArgs()
    >>> args.clip_param
    0.2
    """

    def __init__(self) -> None:
        """
        Initializes the default MAPPO training configuration.

        Parameters
        ----------
        self : TrainerArgs
            The configuration instance.

        Returns
        -------
        None

        Examples
        --------
        >>> args = TrainerArgs()
        >>> args.num_mini_batch
        4
        """
        self.clip_param = 0.2
        self.ppo_epoch = 10
        self.num_mini_batch = 4
        self.data_chunk_length = 4
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.05
        self.max_grad_norm = 0.5
        self.huber_delta = 10.0

        self.use_recurrent_policy = False
        self.use_naive_recurrent_policy = False
        self.use_max_grad_norm = True
        self.use_clipped_value_loss = True
        self.use_huber_loss = False
        self.use_popart = False
        self.use_valuenorm = False
        self.use_value_active_masks = True
        self.use_policy_active_masks = True