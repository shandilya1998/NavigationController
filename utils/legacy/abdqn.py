import torch
import numpy as np
import gym
import stable_baselines3 as sb3
from typing import Any, Dict, List, Optional, Tuple, Type, Union, NamedTuple
from neurorobotics.bg.autoencoder import ResNet18Enc
from neurorobotics.utils.per import SumSegmentTree, MinSegmentTree

TensorDict = Dict[Union[str, int], torch.Tensor]

class PrioritizedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    idx: torch.Tensor
    weights: torch.Tensor

class PrioritizedDictReplayBufferSamples(PrioritizedReplayBufferSamples):
    observations: TensorDict
    actions: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    rewards: torch.Tensor
    idx: torch.Tensor
    weights: torch.Tensor

class FeaturesExtractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, 
        observation_space: gym.Space,
        features_dim: int,
        activation_fn: torch.nn.Module = torch.nn.ReLU,
    ):
        super(FeaturesExtractor, self).__init__(observation_space, features_dim)
        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size = 4, stride = 2, padding = 1),
            activation_fn()
        )

        self.encoder = ResNet18Enc([1, 1, 1, 1], nc = 3, activation_fn = activation_fn)

        self.cnn3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 48, kernel_size = 3, stride = 2, padding = 1),
            activation_fn()
        )

        self.cnn4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size = 5, stride = 2, padding = 1),
            activation_fn(),
            torch.nn.Conv2d(32, 32, kernel_size = 4, stride = 2, padding = 0),
            activation_fn(),
            torch.nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 0),
            activation_fn(),
            torch.nn.Flatten()
        )

        with torch.no_grad():
            t = torch.zeros((1, 64, 30, 30))
            self.n_flatten = self.cnn4(t).size(-1)

        self.output = torch.nn.Sequential(
            torch.nn.Linear(
                self.n_flatten,
                features_dim,
            ),
            activation_fn()
        )

    def forward(self, observations):
        loc_map = observations['loc_map']
        prev_loc_map = observations['prev_loc_map']
        window = observations['window']
        frame_t = observations['frame_t']
        loc_map = self.cnn1(loc_map)
        prev_loc_map = self.cnn1(prev_loc_map)
        features = self.cnn3(torch.cat([loc_map, prev_loc_map], 1))
        features = torch.cat([self.encoder(torch.cat([
            window, frame_t
        ], 1)), features], 1)
        features = self.cnn4(features)
        features = self.output(features)
        return features


class MultiDiscreteQNetwork(sb3.common.policies.BasePolicy):
    """
    Action-Value (Q-Value) network for DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: torch.nn.Module,
        features_dim: int,
        net_arch: Optional[List[List[int]]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        normalize_images: bool = True,
    ):
        super(MultiDiscreteQNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        action_dims = self.action_space.nvec  # number of actions
        if net_arch is None:
            net_arch = [[64, 64]]
            net_arch.extend([[64, 64]] * len(action_dims))
        v_net = sb3.common.torch_layers.create_mlp(self.features_dim, 1, self.net_arch[0], self.activation_fn)
        self.v_net = torch.nn.Sequential(*v_net)
        self.q_nets = torch.nn.ModuleList()
        for i, action_dim in enumerate(action_dims):
            q_net = sb3.common.torch_layers.create_mlp(self.features_dim, action_dim, self.net_arch[i + 1], self.activation_fn)
            self.q_nets.append(torch.nn.Sequential(*q_net))

    def forward(self, obs: torch.Tensor) -> List[torch.Tensor]:
        """
        Predict the q-values.
        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        features = self.extract_features(obs)
        state_value = self.v_net(features)
        q_values = []
        for q_net in self.q_nets:
            out = q_net(features)
            q_values.append(
                out - torch.mean(
                    out,
                    dim = 1,
                    keepdim=True
                ) / out.shape[-1] + state_value
            )

        return q_values

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(observation)
        # Greedy action
        actions = torch.stack(
            [q_value.argmax(dim=1).reshape(-1) for q_value in q_values], -1
        )
        return actions

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class DQNPolicy(sb3.common.policies.BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: sb3.common.type_aliases.Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        features_extractor_class: Type[sb3.common.torch_layers.BaseFeaturesExtractor] = sb3.common.torch_layers.FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(DQNPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is None:
            if features_extractor_class == sb3.common.torch_layers.NatureCNN:
                net_arch = []
            else:
                net_arch = [[64, 64]] * (len(action_space.nvec) + 1)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: sb3.common.type_aliases.Schedule) -> None:
        """
        Create the network and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self) -> MultiDiscreteQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MultiDiscreteQNetwork(**net_args).to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

class PrioritizedReplayBuffer(sb3.common.buffers.ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
    ):
        assert alpha > 0.0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        super(PrioritizedReplayBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination
        )
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos
        )
        self._it_sum[self.pos] = self._max_priority ** self._alpha
        self._it_min[self.pos] = self._max_priority ** self._alpha

    def sample(self, batch_size: int, beta: float, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> PrioritizedReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        assert beta > 0.0
        size = self.buffer_size - 1 if self.full else self.pos
        mass = []
        total = self._it_sum.sum(0, size)
        mass = np.random.random(size=batch_size) * total
        batch_inds = self._it_sum.find_prefixsum_idx(mass)
        return self._get_samples(batch_inds, beta, env=env)

    def _get_samples(self, batch_inds: np.ndarray, beta: float, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> PrioritizedReplayBufferSamples:

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)
        
        size = self.buffer_size if self.full else self.pos + 1
       
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * size) ** (-beta)
        p_sample = self._it_sum[batch_inds] / self._it_sum.sum()
        weights = (p_sample * size) ** (-beta) / max_weight 

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self._normalize_reward(self.rewards[batch_inds], env),
            batch_inds,
            weights
        )
        return PrioritizedReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """

        size = self.buffer_size - 1 if self.full else self.pos
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < size
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha
        self._max_priority = max(self._max_priority, np.max(priorities))


class PrioritizedDictReplayBuffer(sb3.common.buffers.DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
    ):
        assert alpha > 0.0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        super(PrioritizedDictReplayBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination
        )
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos
        )
        self._it_sum[self.pos] = self._max_priority ** self._alpha
        self._it_min[self.pos] = self._max_priority ** self._alpha

    def sample(self, batch_size: int, beta: float, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> PrioritizedDictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        assert beta > 0.0
        size = self.buffer_size - 1 if self.full else self.pos
        mass = []
        total = self._it_sum.sum(0, size)
        mass = np.random.random(size=batch_size) * total
        batch_inds = self._it_sum.find_prefixsum_idx(mass)
        return self._get_samples(batch_inds, beta, env=env)

    def _get_samples(self, batch_inds: np.ndarray, beta: float, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> PrioritizedDictReplayBufferSamples:

        obs_ = self._normalize_obs({key: obs[batch_inds, 0, :] for key, obs in self.observations.items()})
        next_obs_ = self._normalize_obs({key: obs[batch_inds, 0, :] for key, obs in self.next_observations.items()})

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        size = self.buffer_size if self.full else self.pos + 1
       
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * size) ** (-beta)
        p_sample = self._it_sum[batch_inds] / self._it_sum.sum()
        weights = (p_sample * size) ** (-beta) / max_weight 

        return PrioritizedDictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds] * (1 - self.timeouts[batch_inds])),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds], env)),
            idx = self.to_torch(batch_inds),
            weights = self.to_torch(weights)
        )

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """ 
        size = self.buffer_size if self.full else self.pos + 1
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < size
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha
        self._max_priority = max(self._max_priority, np.max(priorities))


class ABDQN(sb3.common.off_policy_algorithm.OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)
    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[sb3.common.type_aliases.GymEnv, str],
        learning_rate: Union[float, sb3.common.type_aliases.Schedule] = 1e-4,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        prioritized_replay_beta: float = 0.4,
        prioritized_replay_eps: float = 1e-6,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[sb3.common.buffers.ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(ABDQN, self).__init__(
            policy,
            env,
            DQNPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.MultiDiscrete,),
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.prioritized_replay_beta = None
        self.prioritized_replay_eps = None
        if replay_buffer_class == PrioritizedReplayBuffer or replay_buffer_class == PrioritizedDictReplayBuffer:
            self.beta = 0.0
            self.beta_schedule = None
            self.prioritized_replay_beta = prioritized_replay_beta
            self.prioritized_replay_eps = prioritized_replay_eps
            self.prioritized_replay = True
        self.q_net, self.q_net_target = None, None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(ABDQN, self)._setup_model()
        self._create_aliases()
        self.exploration_schedule = sb3.common.utils.get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.prioritized_replay:
            assert self.prioritized_replay_beta is not None
            self.beta_schedule = sb3.common.utils.get_linear_fn(
                start = self.prioritized_replay_beta,
                end = 1.0,
                end_fraction = 1.0
            )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            sb3.common.utils.polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        if self.prioritized_replay:
            self.beta = self.beta_schedule(self._current_progress_remaining)
            self.logger.record("rollout/beta", self.beta)
        self.logger.record("rollout/exploration rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = None
            if self.prioritized_replay:
                replay_data = self.replay_buffer.sample(batch_size, beta = self.beta, env=self._vec_normalize_env)
            else:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                temp_q_values = self.q_net(replay_data.next_observations)
                max_actions = torch.stack([torch.argmax(q_val, dim=1) for q_val in temp_q_values], -1)
                next_q_values = sum([
                    torch.gather(
                        q_val,
                        dim=1,
                        index=max_actions[:, i].unsqueeze(-1)
                    ) for i, q_val in enumerate(next_q_values)
                ]) / len(next_q_values)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = sum([
                torch.gather(
                    q_val,
                    dim=1,
                    index=replay_data.actions[:, i].long().unsqueeze(-1)
                ) for i, q_val in enumerate(current_q_values)
            ]) / len(current_q_values)

            # Compute Huber loss (less sensitive to outliers)
            delta = torch.abs(target_q_values - current_q_values)
            threshold = 1.0
            mask = delta < threshold
            loss = (0.5 * mask * (delta ** 2)) + ~mask * (threshold * (delta - 0.5 * threshold))

            if self.prioritized_replay:
                loss = torch.mean(loss * replay_data.weights)
                priorities = delta.squeeze(-1) + self.prioritized_replay_eps
                self.replay_buffer.update_priorities(replay_data.idx.cpu().detach().numpy(), priorities.cpu().detach().numpy())
            else:
                loss = torch.mean(loss)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.
        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if sb3.common.utils.is_vectorized_observation(sb3.common.preprocessing.maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: sb3.common.type_aliases.MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[sb3.common.type_aliases.GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> sb3.common.off_policy_algorithm.OffPolicyAlgorithm:

        return super(ABDQN, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(ABDQN, self)._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
