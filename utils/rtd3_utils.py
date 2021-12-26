import stable_baselines3 as sb3
import gym
import numpy as np
from typing import NamedTuple, Any, Dict, List, Optional, Tuple, Type, Union
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
import math
import torch
import copy
import warnings

TensorDict = Dict[Union[str, int], torch.Tensor]

class ReplayBufferSamples(NamedTuple):
    observations: List[torch.Tensor]
    actions: List[torch.Tensor]
    next_observations: List[torch.Tensor]
    dones: List[torch.Tensor]
    rewards: List[torch.Tensor]


class DictReplayBufferSamples(ReplayBufferSamples):
    observations: List[TensorDict]
    actions: List[torch.Tensor]
    next_observations: List[torch.Tensor]
    dones: List[torch.Tensor]
    rewards: List[torch.Tensor]

class EpisodicReplayBuffer(sb3.common.buffers.BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        max_episode_size: int = 250,
    ):
        self.n_ep = int(buffer_size / max_episode_size)
        self.max_ep_size = max_episode_size
        super(EpisodicReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.observations = np.zeros((self.n_ep, self.max_ep_size + 1, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
            self.next_observations = None
        else:
            self.observations = np.zeros((self.n_ep, self.max_ep_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
            self.next_observations = np.zeros((self.n_ep, self.max_ep_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.actions = np.zeros((self.n_ep, self.max_ep_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.n_ep, self.max_ep_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_ep, self.max_ep_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.n_ep, self.max_ep_size, self.n_envs), dtype=np.float32)
        self.episode_lengths = np.zeros((self.n_ep, self.n_envs), dtype=np.int32) 
        self.ep = 0
        self.pos = np.zeros((self.n_envs), dtype = np.int32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes + self.episode_lengths.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
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
        # Copy to avoid modification by reference
        self.observations[self.ep, self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[self.ep, self.pos + 1] = np.array(next_obs).copy()
        else:
            self.next_observations[self.ep, self.pos] = np.array(next_obs).copy()

        self.actions[self.ep, self.pos] = np.array(action).copy()
        self.rewards[self.ep, self.pos] = np.array(reward).copy()
        self.dones[self.ep, self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.ep, self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        for i, d in enumerate(done):
            if d:
                self.episode_lengths[self.ep, i] = self.pos + 1
                self.pos[i] = 0
                self.ep += 1
            else:
                self.pos[i] += 1
            if i >= 1:
                raise ValueError("Replay buffer only support single environment for now")
        if self.ep == self.n_ep:
            self.full = True
            self.ep = 0

    def sample(self, batch_size: int, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> sb3.common.type_aliases.ReplayBufferSamples:
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
        batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> sb3.common.type_aliases.ReplayBufferSamples:
        """
            Need to update this.
            Refer to EpisodicDictReplayBuffer
        """
        raise NotImplementedError
        size = self.episode_lengths[batch_inds].max()
        if self.optimize_memory_usage:
            obs = self._normalize_obs(self.observations[batch_inds, :-1, 0, :], env)
            next_obs = self._normalize_obs(self.observations[batch_inds, 1: 0, :], env)
        else:
            obs = self._normalize_obs(self.observations[batch_inds, :, 0, :], env)
            next_obs = self._normalize_obs(self.next_observations[batch_inds, :, 0, :], env)
        obs = obs[:, :size]
        next_obs = next_obs[:, :size]
        actions = self.actions[batch_inds, :size, 0, :]
        dones = self.dones[batch_inds, :size] * (1 - self.timeouts[batch_inds, :size])
        rewards = self._normalize_reward(self.rewards[batch_inds, :size], env)

        data = (
            obs,
            actions,
            next_obs,
            dones,
            rewards,
        )
        return sb3.common.type_aliases.ReplayBufferSamples(*tuple(map(self.to_torch, data)))

class EpisodicDictReplayBuffer(sb3.common.buffers.BaseBuffer):
    """ 
    Replay buffer used in off-policy algorithms like SAC/TD3.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        max_episode_size: int = 250,
    ):

        self.n_ep = int(buffer_size / max_episode_size)
        self.max_ep_size = max_episode_size
        super(EpisodicDictReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        assert n_envs == 1, "Replay buffer only support single environment for now"
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.n_ep, self.max_ep_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.n_ep, self.max_ep_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        # only 1 env is supported
        self.actions = np.zeros((self.n_ep, self.max_ep_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.n_ep, self.max_ep_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_ep, self.max_ep_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.n_ep, self.max_ep_size, self.n_envs), dtype=np.float32)
        self.episode_lengths = np.zeros((self.n_ep, self.n_envs), dtype=np.int32)
        self.ep = 0
        self.pos = np.zeros((self.n_envs), dtype = np.int32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes + self.episode_lengths.nbytes
            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            self.observations[key][self.ep, self.pos] = np.array(obs[key]).copy()

        for key in self.next_observations.keys():
            self.next_observations[key][self.ep, self.pos] = np.array(next_obs[key]).copy()

        self.actions[self.ep, self.pos] = np.array(action).copy()
        self.rewards[self.ep, self.pos] = np.array(reward).copy()
        self.dones[self.ep, self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.ep, self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        for i, d in enumerate(done):
            if d or self.pos[i] >= self.max_ep_size - 1:
                self.episode_lengths[self.ep, i] = self.pos[i] + 1
                self.pos[i] = 0
                self.ep += 1
            else:
                self.pos[i] += 1
            if i >= 1:
                raise ValueError("Replay buffer only support single environment for now")
        if self.ep == self.n_ep:
            self.full = True
            self.ep = 0

    def sample(self, batch_size: int, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> DictReplayBufferSamples:
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
        if not self.full:
            batch_inds = np.random.randint(0, self.ep, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.n_ep, size=batch_size)    
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> DictReplayBufferSamples:
        size = int(self.episode_lengths[batch_inds].mean())
        obs = [
            self._normalize_obs({key: obs[batch_inds, i, 0, :] for key, obs in self.observations.items()}) \
                for i in range(size)
        ]
        next_obs = [
            self._normalize_obs({key: obs[batch_inds, i, 0, :] for key, obs in self.next_observations.items()}) \
                for i in range(size)
        ]
        observations = [{key: self.to_torch(ob) for key, ob in obs_.items()} for obs_ in obs]
        next_observations = [{key: self.to_torch(ob) for key, ob in obs_.items()} for obs_ in next_obs]
        actions = [
            self.to_torch(
                self.actions[batch_inds, i, 0, :]
            ) for i in range(size)
        ]
        dones = [
            self.to_torch(
                self.dones[batch_inds, i] * (1 - self.timeouts[batch_inds, i])
            ) for i in range(size)
        ]
        rewards = [
            self.to_torch(
                self._normalize_reward(self.rewards[batch_inds, i], env)
            ) for i in range(size)
        ]
        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones = dones,
            rewards = rewards
        )

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, output_dim, net_arch, squash_output = True):
        super(LSTM, self).__init__()
        if len(net_arch) > 0: 
            self.layers = [torch.nn.LSTMCell(input_dim, net_arch[0])]
        else:
            self.layers = [] 

        for idx in range(len(net_arch) - 1):
            self.layers.append(torch.nn.LSTMCell(net_arch[idx], net_arch[idx + 1])) 

        if output_dim > 0: 
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
            self.layers.append(torch.nn.Linear(last_layer_dim, output_dim))
        if squash_output:
            self.layers.append(torch.nn.Tanh())
        self.offset = 2 if squash_output else 1
        self.layers = torch.nn.ModuleList(self.layers)
        self.net_arch = net_arch

    def forward(self, x, states):
        next_states = []
        for i in range(len(self.net_arch)):
            state = self.layers[i](x, states[i])
            next_states.append(state)
            x = state[0]
        for i in range(self.offset):
            x = self.layers[-self.offset + i](x)
        return x, next_states

def create_lstm(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    squash_output: bool = False,
) -> List[torch.nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [torch.nn.LSTMCell(input_dim, net_arch[0])]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(torch.nn.LSTMCell(net_arch[idx], net_arch[idx + 1]))

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(torch.nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(torch.nn.Tanh())
    return modules

class RecurrentActor(sb3.common.policies.BasePolicy):
    """
    Actor network (policy) for TD3.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a torch.nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: torch.nn.Module,
        features_dim: int,
        normalize_images: bool = True,
    ):
        super(RecurrentActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim

        action_dim = sb3.common.preprocessing.get_action_dim(self.action_space)
        self.mu = LSTM(features_dim, action_dim, net_arch, squash_output=True)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: torch.Tensor, state: List[Tuple[torch.Tensor]]) -> torch.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        return self.mu(features, state)

    def _predict(self, observation: torch.Tensor, state: List[Tuple[torch.Tensor]], deterministic: bool = False) -> torch.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.forward(observation, state)
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Union[List[Tuple[torch.Tensor]], List[torch.Tensor]] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Union[List[Tuple[torch.Tensor]], List[torch.Tensor]]]:
        """ 
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]

        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if sb3.common.preprocessing.is_image_space(obs_space):
                    obs_ = sb3.common.preprocessing.maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or sb3.common.utils.is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif sb3.common.preprocessing.is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = sb3.common.preprocessing.maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = sb3.common.utils.is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = sb3.common.utils.obs_as_tensor(observation, self.device)

        with torch.no_grad():
            actions, state = self._predict(observation, state, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, state

class RecurrentContinuousCritic(sb3.common.policies.BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.
    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: torch.nn.Module,
        features_dim: int,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = sb3.common.preprocessing.get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = LSTM(features_dim + action_dim, 1, net_arch, squash_output = False)
            #q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, states: List[List[Tuple[torch.Tensor]]], actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = torch.cat([features, actions], dim=1)
        _states = []
        outputs = []
        for i, q_net in enumerate(self.q_networks):
            out, state = q_net(qvalue_input, states[i])
            outputs.append(out)
            _states.append(state)
        return tuple(outputs), _states

    def q1_forward(self, obs: torch.Tensor, states: List[Tuple[torch.Tensor]], actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](torch.cat([features, actions], dim=1), states)

class RecurrentTD3Policy(sb3.common.policies.BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.
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
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: sb3.common.type_aliases.Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        features_extractor_class: Type[sb3.common.torch_layers.BaseFeaturesExtractor] = sb3.common.torch_layers.BaseFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(RecurrentTD3Policy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [400, 300]

        actor_arch, critic_arch = sb3.common.torch_layers.get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
                "activation_fn": self.activation_fn
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: sb3.common.type_aliases.Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extactor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> RecurrentActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return RecurrentActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> sb3.common.policies.ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return RecurrentContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: torch.Tensor, state: List[Tuple[torch.Tensor]], deterministic: bool = False) -> torch.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: torch.Tensor, state: List[Tuple[torch.Tensor]], deterministic: bool = False) -> torch.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation, state)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Union[List[Tuple[torch.Tensor]], List[torch.Tensor]] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Union[List[Tuple[torch.Tensor]], List[torch.Tensor]]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]

        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if sb3.common.preprocessing.is_image_space(obs_space):
                    obs_ = sb3.common.preprocessing.maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or sb3.common.utils.is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif sb3.common.preprocessing.is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = sb3.common.preprocessing.maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = sb3.common.utils.is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = sb3.common.utils.obs_as_tensor(observation, self.device)

        with torch.no_grad():
            actions, state = self._predict(observation, state, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, state

class RTD3(sb3.common.off_policy_algorithm.OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.
    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
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
        policy: Union[str, Type[RecurrentTD3Policy]],
        env: Union[sb3.common.type_aliases.GymEnv, str],
        learning_rate: Union[float, sb3.common.type_aliases.Schedule] = 1e-3,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[sb3.common.noise.ActionNoise] = None,
        replay_buffer_class: Optional[sb3.common.buffers.ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        n_steps: int = 5
    ):
        self.n_steps = n_steps
        super(RTD3, self).__init__(
            policy,
            env,
            RecurrentTD3Policy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
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
            supported_action_spaces=(gym.spaces.Box),
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(RTD3, self)._setup_model()
        self.hidden_state = [
            (torch.zeros((1, size)).to(self.device), torch.zeros((1, size)).to(self.device)) \
                for size in self.policy.net_arch
        ]
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.n_critics = self.policy.critic_kwargs['n_critics']
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[sb3.common.noise.ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, self.hidden_state = self.predict(self._last_obs, self.hidden_state, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _store_transition(
        self,
        replay_buffer: EpisodicReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).
        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when done is True)
        :param reward: reward for the current transition
        :param done: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            # VecNormalize normalizes the terminal observation
            if self._vec_normalize_env is not None:
                next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
        else:
            next_obs = new_obs_

        if done:
            self.hidden_state = [
                (torch.zeros((1, size)).to(self.device), torch.zeros((1, size)).to(self.device)) \
                    for size in self.policy.net_arch
            ]

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            done,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def update_policy(self,
        gradient_steps: int,
        batch_size: int,
        replay_data: sb3.common.type_aliases.ReplayBufferSamples
    ):
        # Need to modify
        actor_losses, critic_losses = [], []
        hidden_state = [
            (torch.zeros((batch_size, size)).to(self.device), torch.zeros((batch_size, size)).to(self.device)) \
                for size in self.policy.net_arch
        ] 
        hidden_state_critic = [
            [
                (torch.zeros((batch_size, size)).to(self.device), torch.zeros((batch_size, size)).to(self.device)) \
                    for size in self.policy.net_arch
            ] for i in range(self.n_critics)
        ]
        hidden_state_loss = [
            (torch.zeros((batch_size, size)).to(self.device), torch.zeros((batch_size, size)).to(self.device)) \
                for size in self.policy.net_arch
        ] 
        with torch.no_grad():
            next_ac, next_hidden_state = self.actor_target(replay_data.observations[0], hidden_state)
            _, next_hidden_state_critic = self.critic_target(replay_data.observations[0], hidden_state_critic, next_ac)
        num_updates = math.ceil(gradient_steps / self.n_steps)
        for i in range(gradient_steps):
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions[i].clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions, next_hidden_state = self.actor_target(replay_data.next_observations[i], next_hidden_state)
                next_actions = (next_actions + noise).clamp(-1, 1)
                # Compute the next Q-values: min over all critics targets
                next_q_values, next_hidden_state_critic = self.critic_target(replay_data.next_observations[i], next_hidden_state_critic, next_actions)
                next_q_values = torch.cat(next_q_values, dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards[i] + (1 - replay_data.dones[i]) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values, hidden_state_critic = self.critic(replay_data.observations[i], hidden_state_critic, replay_data.actions[i])

            # Compute critic loss
            critic_loss = sum([torch.nn.functional.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            lst_1 = []
            for k in range(len(self.policy.net_arch)):
                lst_1.append(
                    (
                        next_hidden_state[k][0].detach(),
                        next_hidden_state[k][1].detach()
                    )
                )
            next_hidden_state = lst_1
            lst_2 = []
            for k in range(self.n_critics):
                lst_3 = []
                for l in range(len(self.policy.net_arch)):
                    lst_3.append(
                        (
                            hidden_state_critic[k][l][0].detach(),
                            hidden_state_critic[k][l][1].detach()
                        )
                    )
                lst_2.append(lst_3)
            hidden_state_critic = lst_2
            self._n_updates += 1
            actions, hidden_state = self.actor(replay_data.observations[i], hidden_state)
            q_val, hidden_state_loss = self.critic.q1_forward(replay_data.observations[i], hidden_state_loss, actions)
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -q_val.mean()
                actor_losses.append(actor_loss.item())
                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                sb3.common.utils.polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                sb3.common.utils.polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            lst_4 = [] 
            for k in range(len(self.policy.net_arch)):
                lst_4.append(
                    (
                        hidden_state[k][0].detach(),
                        hidden_state[k][1].detach()
                    )
                )
            hidden_state = lst_4
            lst_5 = []
            for k in range(len(self.policy.net_arch)):
                lst_5.append(
                    (
                        hidden_state_loss[k][0].detach(),
                        hidden_state_loss[k][1].detach()
                    )
                )
            hidden_state_loss = lst_5

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0: 
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        remaining = gradient_steps
        while remaining > 0:
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            steps = len(replay_data.dones)
            if remaining > steps:
                remaining -= steps
            else:
                steps = copy.deepcopy(remaining)
                remaining = 0
            self.update_policy(steps, batch_size, replay_data)

    def learn(
        self,
        total_timesteps: int,
        callback: sb3.common.type_aliases.MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[sb3.common.type_aliases.GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TD3",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> sb3.common.off_policy_algorithm.OffPolicyAlgorithm:

        return super(RTD3, self).learn(
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
        return super(RTD3, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
