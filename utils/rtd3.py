import enum
import warnings
import numpy as np
import gym
from numpy.core.fromnumeric import mean
import stable_baselines3 as sb3
from typing import NamedTuple, Any, Dict, List, Optional, Tuple, Union, Type
import torch
import psutil
import copy
from bg.autoencoder import ResNet18Enc
from constants import params
from pytorch_msssim import ssim
from torch.utils.tensorboard import SummaryWriter
import cv2
import os
from utils.td3 import Actor, ContinuousCritic
from stable_baselines3.common.buffers import DictReplayBuffer
"""
Idea of burn in comes from the following paper:
https://openreview.net/pdf?id=r1lyTjAqYX
"""

TensorDict = Dict[Union[str, int], torch.Tensor]
TensorList = List[torch.Tensor]
TensorTuple = Tuple[torch.Tensor, torch.Tensor]


class LSTMState(NamedTuple):
    state: TensorTuple


class RecurrentDictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    rewards: torch.Tensor
    states: TensorList
    next_states: TensorList


class TimeDistributedFeaturesExtractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self,
        observation_space: gym.Space,
        features_dim: int,
        activation_fn: torch.nn.Module = torch.nn.ReLU,
        pretrained_params_path = None,
        device = None,
    ):
        super(TimeDistributedFeaturesExtractor, self).__init__(observation_space, features_dim)

        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size = 4, stride = 2, padding = 1),
            activation_fn()
        )
        self.cnn2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size = 4, stride = 1, padding = 1),
            activation_fn()
        )

        self.encoder = ResNet18Enc([1, 1, 1, 1], nc = 3, activation_fn = activation_fn)

        self.cnn3 = torch.nn.Sequential(
            torch.nn.Conv2d(24, 48, kernel_size = 3, stride = 2, padding = 1),
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
        self.lstm = torch.nn.LSTM(
            self.n_flatten,
            features_dim,
            batch_first = True
        )

    def forward(self, observations, state):
        batch_size = observations['scale_1'].size(0)
        loc_map = observations['loc_map'].contiguous().view(-1, *self._observation_space['loc_map'].shape)
        ego_map = observations['ego_map'].contiguous().view(-1, *self._observation_space['ego_map'].shape)
        scale_1 = observations['scale_1'].contiguous().view(-1, *self._observation_space['scale_1'].shape)
        scale_2 = observations['scale_2'].contiguous().view(-1, *self._observation_space['scale_2'].shape)
        loc_map = self.cnn1(loc_map)
        ego_map = self.cnn2(ego_map)
        features = self.cnn3(torch.cat([loc_map, ego_map], 1))
        features = torch.cat([self.encoder(torch.cat([
            scale_1, scale_2
        ], 1)), features], 1)
        features = self.cnn4(features)
        features = features.view(batch_size, -1, self.n_flatten)
        features, state = self.lstm(features, state)
        features = torch.mean(features, 1)
        return features, state


class DictReplayBuffer(sb3.common.buffers.ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        max_seq_len: int,
        state_spec: Tuple[int, int],
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        assert state_spec is not None
        assert max_seq_len > 2
        self.state_spec = state_spec
        self.max_seq_len = max_seq_len
        super(sb3.common.buffers.ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        
        num_lstm_layers = self.state_spec[0]
        state_size = self.state_spec[1]

        self.states = [
            np.zeros(
                (
                    self.buffer_size,
                    num_lstm_layers,
                    self.n_envs,
                    state_size
                ),
                dtype = np.float32
            ),
            np.zeros(
                (
                    self.buffer_size,
                    num_lstm_layers,
                    self.n_envs,
                    state_size
                ),
                dtype = np.float32
            )
        ]
        self.next_states = [
            np.zeros(
                (
                    self.buffer_size,
                    num_lstm_layers,
                    self.n_envs,
                    state_size
                ),
                dtype = np.float32
            ),
            np.zeros(
                (
                    self.buffer_size,
                    num_lstm_layers,
                    self.n_envs,
                    state_size
                ),
                dtype = np.float32
            )
        ]

        # only 1 env is supported
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
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
        states: Tuple[np.ndarray, np.ndarray],
        next_states: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            self.observations[key][self.pos] = np.array(obs[key]).copy()

        for key in self.next_observations.keys():
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        for i, state in enumerate(states):
            self.states[i][self.pos] = np.array(state).copy()
        for i, state in enumerate(next_states):
            self.next_states[i][self.pos] = np.array(state).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> RecurrentDictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(sb3.common.buffers.ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> RecurrentDictReplayBufferSamples:

        # Computing all indices and zeroing in states and observations for sequence truncation
        offsets = np.repeat(np.expand_dims(np.arange(-(self.max_seq_len - 1), 1), 0), len(batch_inds), 0)
        inds = np.repeat(np.expand_dims(batch_inds, 1), self.max_seq_len, 1) + offsets
        offset = self.buffer_size if self.full else self.pos
        inds[inds < 0] = inds[inds < 0] + offset 
        
        dones = self.dones[inds, 0]
        include = np.flip(np.multiply.accumulate(np.flip(1 - dones, 1), 1), 1)
        
        obs = self._normalize_obs({
            key: obs[inds, 0, :] * self.__expand_include(
                include,
                obs[inds, 0, :].shape
            ).astype(obs.dtype) for key, obs in self.observations.items()
        })
        next_obs = self._normalize_obs({
            key: obs[inds, 0, :] * self.__expand_include(
                include,
                obs[inds, 0, :].shape
            ).astype(obs.dtype) for key, obs in self.next_observations.items()
        })
        dones = self.dones[batch_inds] * (1 - self.timeouts[batch_inds])
        actions = self.actions[batch_inds]
        rewards = self._normalize_reward(self.rewards[batch_inds], env)
        state_include = np.prod(include, 1)
        next_state_include = np.prod(include[:, 1:], 1)
        states = [self.states[i][inds[:, 0], :, 0] * self.__expand_include(
            state_include,
            self.states[i][inds[:, 0], :, 0].shape,
        ).astype(self.states[i].dtype) for i in range(len(self.states))]
        next_states = [self.next_states[i][inds[:, 0], :, 0] * self.__expand_include(
            next_state_include,
            self.next_states[i][inds[:, 0], :, 0].shape,
        ).astype(self.next_states[i].dtype) for i in range(len(self.next_states))]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}
        actions = self.to_torch(actions)
        rewards = self.to_torch(rewards)
        dones = self.to_torch(dones)
        states = [self.to_torch(state).transpose(1, 0) for state in states]
        next_states = [self.to_torch(next_state).transpose(1, 0) for next_state in next_states]

        return RecurrentDictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            states=states,
            next_states=next_states
        )

    def __expand_include(self, include, shape):
        length = len(include.shape)
        assert shape[:length] == include.shape
        shape = shape[length:]
        for item in shape:
            include = np.repeat(np.expand_dims(include, -1), item, -1)
        return include


class ReplayBuffer(sb3.common.buffers.ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        max_seq_len: int,
        state_spec: Tuple[int, int],
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        assert state_spec is not None
        assert max_seq_len > 2
        self.state_spec = state_spec
        self.max_seq_len = max_seq_len
        super(sb3.common.buffers.ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        
        num_lstm_layers = self.state_spec[0]
        state_size = self.state_spec[1]

        self.states = [
            np.zeros(
                (
                    self.buffer_size,
                    self.n_envs,
                    num_lstm_layers,
                    state_size
                ),
                dtype = np.float32
            )
        ] * 2
        self.next_states = [
            np.zeros(
                (
                    self.buffer_size,
                    self.n_envs,
                    num_lstm_layers,
                    state_size
                ),
                dtype = np.float32
            )
        ] * 2

        # only 1 env is supported
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
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
        states: Tuple[np.ndarray, np.ndarray],
        next_states: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            self.observations[key][self.pos] = np.array(obs[key]).copy()

        for key in self.next_observations.keys():
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        for i, state in enumerate(states):
            self.states[i][self.pos] = np.array(state).copy()
        for i, state in enumerate(next_states):
            self.next_states[i][self.pos] = np.array(state).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> RecurrentDictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(sb3.common.buffers.ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> RecurrentDictReplayBufferSamples:

        # Computing all indices and zeroing in states and observations for sequence truncation
        offsets = np.repeat(np.expand_dims(np.arange(-(self.max_seq_len - 1), 1), 0), len(batch_inds), 0)
        inds = np.repeat(np.expand_dims(batch_inds, 1), self.max_seq_len, 1) + offsets
        offset = self.buffer_size if self.full else self.pos
        inds[inds < 0] = inds[inds < 0] + offset 
        
        dones = self.dones[inds, 0]
        include = np.flip(np.multiply.accumulate(np.flip(1 - dones, 1), 1), 1)
        
        obs = self._normalize_obs({
            key: obs[inds, 0, :] * self.__expand_include(
                include,
                obs[inds, 0, :].shape
            ).astype(obs.dtype) for key, obs in self.observations.items()
        })
        next_obs = self._normalize_obs({
            key: obs[inds, 0, :] * self.__expand_include(
                include,
                obs[inds, 0, :].shape
            ).astype(obs.dtype) for key, obs in self.next_observations.items()
        })
        dones = self.dones[batch_inds] * (1 - self.timeouts[batch_inds])
        actions = self.actions[batch_inds]
        rewards = self._normalize_reward(self.rewards[batch_inds], env)
        state_include = np.prod(include, 1)
        next_state_include = np.prod(include[:, 1:], 1)
        states = [self.states[i][inds[:, 0], 0, :] * self.__expand_include(
            state_include,
            self.states[i][inds[:, 0], 0, :].shape,
        ).astype(self.states[i].dtype) for i in range(len(self.states))]
        next_states = [self.next_states[i][inds[:, 0], 0, :] * self.__expand_include(
            next_state_include,
            self.next_states[i][inds[:, 0], 0, :].shape,
        ).astype(self.next_states[i].dtype) for i in range(len(self.next_states))]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}
        actions = self.to_torch(actions)
        rewards = self.to_torch(rewards)
        dones = self.to_torch(dones)
        states = [self.to_torch(state).transpose(1, 0) for state in states]
        next_states = [self.to_torch(next_state).transpose(1, 0) for next_state in next_states]

        return RecurrentDictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            states=states,
            next_states=next_states
        )

    def __expand_include(self, include, shape):
        length = len(include.shape)
        assert shape[:length] == include.shape
        shape = shape[length:]
        for item in shape:
            include = np.repeat(np.expand_dims(include, -1), item, -1)
        return include


class HerReplayBuffer(sb3.common.buffers.ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        max_seq_len: int,
        state_spec: Tuple[int, int],
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        assert state_spec is not None
        assert max_seq_len > 2
        self.state_spec = state_spec
        self.max_seq_len = max_seq_len
        super(sb3.common.buffers.ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        
        num_lstm_layers = self.state_spec[0]
        state_size = self.state_spec[1]

        self.states = [
            np.zeros(
                (
                    self.buffer_size,
                    self.n_envs,
                    num_lstm_layers,
                    state_size
                ),
                dtype = np.float32
            )
        ] * 2
        self.next_states = [
            np.zeros(
                (
                    self.buffer_size,
                    self.n_envs,
                    num_lstm_layers,
                    state_size
                ),
                dtype = np.float32
            )
        ] * 2

        # only 1 env is supported
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
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
        states: Tuple[np.ndarray, np.ndarray],
        next_states: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            self.observations[key][self.pos] = np.array(obs[key]).copy()

        for key in self.next_observations.keys():
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        for i, state in enumerate(states):
            self.states[i][self.pos] = np.array(state).copy()
        for i, state in enumerate(next_states):
            self.next_states[i][self.pos] = np.array(state).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> RecurrentDictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(sb3.common.buffers.ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> RecurrentDictReplayBufferSamples:

        # Computing all indices and zeroing in states and observations for sequence truncation
        offsets = np.repeat(np.expand_dims(np.arange(-(self.max_seq_len - 1), 1), 0), len(batch_inds), 0)
        inds = np.repeat(np.expand_dims(batch_inds, 1), self.max_seq_len, 1) + offsets
        offset = self.buffer_size if self.full else self.pos
        inds[inds < 0] = inds[inds < 0] + offset 
        
        dones = self.dones[inds, 0]
        include = np.flip(np.multiply.accumulate(np.flip(1 - dones, 1), 1), 1)
        
        obs = self._normalize_obs({
            key: obs[inds, 0, :] * self.__expand_include(
                include,
                obs[inds, 0, :].shape
            ).astype(obs.dtype) for key, obs in self.observations.items()
        })
        next_obs = self._normalize_obs({
            key: obs[inds, 0, :] * self.__expand_include(
                include,
                obs[inds, 0, :].shape
            ).astype(obs.dtype) for key, obs in self.next_observations.items()
        })
        dones = self.dones[batch_inds] * (1 - self.timeouts[batch_inds])
        actions = self.actions[batch_inds]
        rewards = self._normalize_reward(self.rewards[batch_inds], env)
        state_include = np.prod(include, 1)
        next_state_include = np.prod(include[:, 1:], 1)
        states = [self.states[i][inds[:, 0], 0, :] * self.__expand_include(
            state_include,
            self.states[i][inds[:, 0], 0, :].shape,
        ).astype(self.states[i].dtype) for i in range(len(self.states))]
        next_states = [self.next_states[i][inds[:, 0], 0, :] * self.__expand_include(
            next_state_include,
            self.next_states[i][inds[:, 0], 0, :].shape,
        ).astype(self.next_states[i].dtype) for i in range(len(self.next_states))]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}
        actions = self.to_torch(actions)
        rewards = self.to_torch(rewards)
        dones = self.to_torch(dones)
        states = [self.to_torch(state).transpose(1, 0) for state in states]
        next_states = [self.to_torch(next_state).transpose(1, 0) for next_state in next_states]

        return RecurrentDictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            states=states,
            next_states=next_states
        )

    def __expand_include(self, include, shape):
        length = len(include.shape)
        assert shape[:length] == include.shape
        shape = shape[length:]
        for item in shape:
            include = np.repeat(np.expand_dims(include, -1), item, -1)
        return include


class Actor(sb3.td3.policies.Actor):
    """
    Actor network (policy) for TD3.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a torch.nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
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
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        normalize_images: bool = True,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images
        )

    def extract_features(self,
            obs: (torch.Tensor, TensorDict),
            state: TensorTuple
        ) -> Tuple[torch.Tensor, TensorTuple]:
        """
        Preprocess the observation if needed and extract features.
        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = sb3.common.preprocessing.preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs, state)

    def forward(self, obs: Union[torch.Tensor, TensorDict], state: TensorTuple) -> Tuple[torch.Tensor, TensorTuple]:
        features, state = self.extract_features(obs, state)
        out = self.mu(features)
        return out, state

    def _predict(self, obs: Union[torch.Tensor, TensorDict], state: TensorTuple) -> Tuple[torch.Tensor, TensorTuple]:
        return self.forward(obs, state)
 
    def predict(self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: List[np.ndarray],
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False):

        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, _obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                obs = _obs[:, 0]
                if sb3.common.preprocessing.is_image_space(obs_space):
                    obs_ = sb3.common.preprocessing.maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or sb3.common.utils.is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key][0] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif sb3.common.preprocessing.is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = np.split(observation, steps, 1)
            observation[0][:, 0] = sb3.common.preprocessing.maybe_transpose(observation[step][:, 0], self.observation_space)
            observation = np.concatenate(observation, 1)
        else:
            observation = np.array(observation)

        observation = sb3.common.utils.obs_as_tensor(observation, self.device)
        state = [torch.as_tensor(s).to(self.device) for s in state]

        with torch.no_grad():
            actions, state = self._predict(observation, state, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()
        state = [s.cpu().numpy() for s in state]

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
            gen_image = gen_image[0]
            depth = depth[0]
            state = [s[:, 0] for s in state]

        return actions, state


class ContinuousCritic(sb3.common.policies.ContinuousCritic):
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
        (a CNN when using images, a torch.nn.Flatten() layer otherwise)
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
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor
        )


    def extract_features(self,
            obs: Union[torch.Tensor, TensorDict],
            state: TensorTuple
        ) -> Tuple[torch.Tensor, TensorTuple]:
        """
        Preprocess the observation if needed and extract features.
        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = sb3.common.preprocessing.preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs, state)

    def forward(self,
            obs: Union[torch.tensor, TensorDict],
            actions: torch.tensor,
            state: TensorTuple
        ) -> Tuple[torch.tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features, _ = self.extract_features(obs, state)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self,
            obs: Union[torch.tensor, TensorDict],
            actions: torch.tensor,
            state: TensorTuple
        ) -> Tuple[torch.tensor, ...]:
        with torch.no_grad():
            features, _ = self.extract_features(obs, state)
        return self.q_networks[0](torch.cat([features, actions], dim=1))


class RTD3Policy(sb3.td3.policies.TD3Policy):
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
        activation_fn: Type[torch.nn.Module] = torch.nn.Tanh,
        features_extractor_class: Type[sb3.common.torch_layers.BaseFeaturesExtractor] = sb3.common.torch_layers.FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(RTD3Policy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor
        )


    def make_actor(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self,
            observation: Union[torch.Tensor, TensorDict],
            state: TensorTuple,
            deterministic: bool = False
        ) -> Tuple[torch.Tensor, TensorTuple]:
        return self._predict(observation, state, deterministic=deterministic)

    def _predict(self,
            observation: Union[torch.Tensor, TensorDict],
            state: TensorTuple,
            deterministic: Optional[bool] = False
        ) -> Tuple[torch.Tensor, TensorTuple]:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation, state)

    def predict(self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: List[np.ndarray],
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False):

        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, _obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                obs = _obs[:, 0]
                if sb3.common.preprocessing.is_image_space(obs_space):
                    obs_ = sb3.common.preprocessing.maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or sb3.common.utils.is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key][0] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif sb3.common.preprocessing.is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = np.split(observation, steps, 1)
            observation[0][:, 0] = sb3.common.preprocessing.maybe_transpose(observation[step][:, 0], self.observation_space)
            observation = np.concatenate(observation, 1)
        else:
            observation = np.array(observation)

        observation = sb3.common.utils.obs_as_tensor(observation, self.device)
        state = [torch.as_tensor(s).to(self.device) for s in state]

        with torch.no_grad():
            actions, state = self._predict(observation, state, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()
        state = [s.cpu().numpy() for s in state]

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
            gen_image = gen_image[0]
            depth = depth[0]
            state = [s[:, 0] for s in state]

        return actions, state


class RTD3(sb3.TD3):
    def __init__(
        self,
        policy: Union[str, Type[RTD3Policy]],
        env: Union[sb3.common.type_aliases.GymEnv, str],
        learning_rate: Union[float, sb3.common.type_aliases.Schedule] = 1e-3,
        buffer_size: int = 1000000,  # 1e6
        max_seq_len: int = 10,
        state_spec: Tuple[int, int] = (1, 512),
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
    ):
        self.max_seq_len = max_seq_len
        self.state_spec = state_spec
        super(RTD3, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            policy_delay,
            target_policy_noise,
            target_noise_clip,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use DictReplayBuffer if needed
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, gym.spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        elif self.replay_buffer_class == HerReplayBuffer:
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"

            # If using offline sampling, we need a classic replay buffer too
            if self.replay_buffer_kwargs.get("online_sampling", True):
                replay_buffer = None
            else:
                replay_buffer = DictReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_space,
                    self.max_seq_len,
                    self.state_spec,
                    self.device,
                    optimize_memory_usage=self.optimize_memory_usage,
                )

        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.max_seq_len,
                self.state_spec,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
        self._create_aliases()

    def _setup_learn(self,
            total_timesteps: int,
            eval_env: Optional[sb3.common.type_aliases.GymEnv],
            callback: sb3.common.type_aliases.MaybeCallback = None,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run"
        ) -> Tuple[int, sb3.common.callbacks.BaseCallback]:
        total_timesteps, callback =  super()._setup_learn(
                total_timesteps,
                eval_env, callback,
                eval_freq,
                n_eval_episodes,
                log_path,
                reset_num_timesteps,
                tb_log_name)
         
        self._last_state = (
            np.zeros((self.state_spec[0], 1, self.state_spec[1]), dtype = np.float32),
            np.zeros((self.state_spec[0], 1, self.state_spec[1]), dtype = np.float32),    
        )
        self._next_state = (
            np.zeros((self.state_spec[0], 1, self.state_spec[1]), dtype = np.float32),
            np.zeros((self.state_spec[0], 1, self.state_spec[1]), dtype = np.float32),    
        )

        return total_timesteps, callback
    
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation
        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, mask, deterministic)
    
    def __get_obs(self):
        obs = {key: np.expand_dims(item, 1) for key, item in self._last_obs.items()}
        return obs

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
            # Pretraining Phase
            unscaled_action = self._last_obs['sampled_action']
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            obs = self.__get_obs()
            unscaled_action, self._next_state = self.predict(
                obs,
                state = self._last_state,
                deterministic=False
            )
        
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
        replay_buffer: sb3.common.buffers.ReplayBuffer,
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

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            done,
            infos,
            self._last_state,
            self._next_state,
        )

        self._last_state = copy.deepcopy(self._next_state)
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
        self,
        env: sb3.common.vec_env.VecEnv,
        callback: sb3.common.callbacks.BaseCallback,
        train_freq: sb3.common.type_aliases.TrainFreq,
        replay_buffer: sb3.common.buffers.ReplayBuffer,
        action_noise: Optional[sb3.common.noise.ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> sb3.common.type_aliases.RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, sb3.common.vec_env.VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while sb3.common.utils.should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return sb3.common.type_aliases.RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not sb3.common.utils.should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1 
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)
                self._last_state = (
                    np.zeros((self.state_spec[0], 1, self.state_spec[1]), dtype = np.float32),
                    np.zeros((self.state_spec[0], 1, self.state_spec[1]), dtype = np.float32),    
                )
                self._next_state = (
                    np.zeros((self.state_spec[0], 1, self.state_spec[1]), dtype = np.float32),
                    np.zeros((self.state_spec[0], 1, self.state_spec[1]), dtype = np.float32),    
                )
                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return sb3.common.type_aliases.RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        supervised_losses = []
        supervised_loss_ratios = []

        for _ in range(0, gradient_steps, params['max_seq_len']):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if self.num_timesteps > params['imitation_steps']:
                with torch.no_grad():
                    # Select action according to policy and add clipped noise
                    noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions, _ = self.actor_target(replay_data.next_observations, replay_data.next_states)
                    next_actions = (next_actions + noise).clamp(-1, 1)

                    # Compute the next Q-values: min over all critics targets
                    next_q_values = self.critic_target(replay_data.next_observations, next_actions, replay_data.next_states)
                    next_q_values = torch.cat(next_q_values, dim=1)
                    next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates for each critic network
                current_q_values = self.critic(replay_data.observations, replay_data.actions, replay_data.states)

                # Compute critic loss
                critic_loss = sum([torch.nn.functional.l1_loss(current_q, target_q_values) for current_q in current_q_values])
                critic_losses.append(critic_loss.item())

                # Optimize the critics
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                # Delayed policy updates
                # Compute actor loss
                action, _ = self.actor(replay_data.observations, replay_data.states)
                if self.num_timesteps < params['staging_steps'] + params['imitation_steps']:
                    ratio = 1.0 - (self.num_timesteps - params['imitation_steps']) / params['staging_steps']
                    supervised_loss_ratios.append(ratio)
                    supervised_loss = torch.nn.functional.l1_loss(action, replay_data.observations['scaled_sampled_action'][:, -1, :])
                    supervised_losses.append(supervised_loss.item())
                    q, _ = self.critic.q1_forward(replay_data.observations, action, replay_data.states)
                    q_loss = -q.mean()
                    actor_losses.append(q_loss.item())
                    actor_loss = supervised_loss * ratio + q_loss * (1 - ratio)
                else:
                    supervised_loss_ratios.append(0.0)
                    q, _ = self.critic.q1_forward(replay_data.observations, action, replay_data.states)
                    actor_loss = -q.mean()
                    actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                sb3.common.utils.polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                sb3.common.utils.polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

            else:
                action, _ = self.actor(replay_data.observations, replay_data.states)
                supervised_loss_ratios.append(1.0)
                supervised_loss = torch.nn.functional.l1_loss(action, replay_data.observations['scaled_sampled_action'][:, -1, :])
                supervised_losses.append(supervised_loss.item())
                self.actor.optimizer.zero_grad()
                supervised_loss.backward()
                self.actor.optimizer.step()
                sb3.common.utils.polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                sb3.common.utils.polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(critic_losses) > 0:
            self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/supervised_loss_ratio", np.mean(supervised_loss_ratios))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))

        if len(supervised_losses) > 0:
            self.logger.record("train/supervised_loss", np.mean(supervised_losses))


def train_autoencoder(
        logdir,
        env,
        n_epochs,
        batch_size,
        learning_rate,
        save_freq,
        eval_freq,
    ):
    # Setting Training Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model Initialisation
    model = Autoencoder(num_Blocks=[1, 1, 1, 1], z_dim = 2 * params['num_ctx']).to(device)
    
    # Optimiser and Scheduler Initalisation
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, 0.99
    )
    
    from sb3.common.buffers import DictReplayBuffer
    # Buffer Initialisation
    buff = DictReplayBuffer( 
        buffer_size=params['buffer_size'],
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    
    # Initialise Tensorboard logger
    writer = SummaryWriter(log_dir = logdir)


    last_obs = env.reset()
    for i in range(1, n_epochs + 1):
        # Evaluation
        if i % eval_freq == 0 and not i == 0:
            total_reward = 0
            losses = []
            L1 = []
            L1_DEPTH = []
            SSIM_1 = []
            SSIM_2 = []
            SSIM_DEPTH = []
            image_size = (64 * 3, 64 * 2)
            video = cv2.VideoWriter(
                os.path.join(logdir, 'model_{}_evaluation.avi'.format(i)),
                cv2.VideoWriter_fourcc(*"MJPG"), 10, image_size, isColor = True
            )
            steps = 0
            model.eval()
            for _ in range(5):
                last_obs = env.reset()
                done = False
                while not done:
                    obs, reward, done, info = env.step(last_obs['sampled_action'])
                    gt_image = torch.from_numpy(np.concatenate([
                        obs['scale_1'], obs['scale_2']
                    ], 1)).float() / 255
                    gt_depth = torch.from_numpy(obs['depth'])
                    gt_image = gt_image.to(device=device)
                    gt_depth = gt_depth.to(device=device)

                    # Model Evaluation
                    with torch.no_grad():
                        _, [gen_image, depth] = model(gt_image.contiguous())
                    l1_gen_image = torch.nn.functional.l1_loss(gen_image, gt_image).item()
                    l1_depth = torch.nn.functional.l1_loss(depth, gt_depth).item()
                    L1.append(l1_gen_image)
                    L1_DEPTH.append(l1_depth)
                    scale_1, scale_2 = torch.split(gt_image, 3, dim = 1)
                    gen_scale_1, gen_scale_2 = torch.split(gen_image, 3, dim = 1)
                    # SSIM computation
                    ssim_scale_1 = 1 - ssim(
                        scale_1, gen_scale_1,
                        data_range=1, size_average=True
                    ).item()
                    ssim_scale_2 = 1 - ssim(
                        scale_2, gen_scale_2,
                        data_range=1, size_average=True
                    ).item()
                    ssim_depth = 1 - ssim(
                        depth, gt_depth, size_average=True,
                        data_range=1
                    ).item()
                    SSIM_1.append(ssim_scale_1)
                    SSIM_2.append(ssim_scale_2)
                    SSIM_DEPTH.append(ssim_depth)
                    loss = ssim_scale_1 + ssim_scale_2 + ssim_depth + l1_depth + l1_gen_image
                    losses.append(loss)
                    
                    # Sampling last frame for writing to video
                    scale_1 = scale_1[0].cpu().numpy()
                    scale_2 = scale_2[0].cpu().numpy()
                    gt_depth = gt_depth[0].cpu().numpy()
                    gen_scale_1 = gen_scale_1[0].cpu().numpy()
                    gen_scale_2 = gen_scale_2[0].cpu().numpy()
                    depth = depth[0].cpu().numpy()
                    
                    observation = np.concatenate([
                        np.concatenate([
                            scale_1,
                            gen_scale_1
                        ], 1),
                        np.concatenate([
                            scale_2,
                            gen_scale_2
                        ], 1),
                        np.repeat(np.concatenate([
                            gt_depth,
                            depth
                        ], 1), 3, 0),
                    ], 2).transpose(1, 2, 0) * 255 
                    observation = observation.astype(np.uint8)
                    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                    video.write(observation)
                    steps += 1
                    total_reward += reward
                    last_obs = obs

            # Writing Evalulation Metrics to Tensorboard
            total_reward = total_reward / 5
            print('-----------------------------')
            print('Evaluation Total Reward {:.4f} Loss {:.4f} L1 {:.4f} L1 depth {:.4f} SSIM_1 {:.4f} SSIM_2 {:.4f} SSIM_DEPTH {:.4f} Steps {}'.format(
                total_reward[0], np.mean(losses), np.mean(L1), np.mean(L1_DEPTH),
                np.mean(SSIM_1), np.mean(SSIM_2), np.mean(SSIM_DEPTH), steps))
            print('-----------------------------')
            writer.add_scalar('Eval/Loss', np.mean(losses), i)
            writer.add_scalar('Eval/L1', np.mean(L1), i)
            writer.add_scalar('Eval/ssim_1', np.mean(SSIM_1), i)
            writer.add_scalar('Eval/ssim_2', np.mean(SSIM_2), i)
            writer.add_scalar('Eval/ssim_depth', np.mean(SSIM_DEPTH), i)
            writer.add_scalar('Eval/depth', np.mean(L1_DEPTH), i)
            cv2.destroyAllWindows()
            video.release()
            model.train()


        # Data Sampling
        total_reward = 0
        count = 0
        for _ in range(3):
            last_obs = env.reset()
            done = False
            while not done:
                obs, reward, done, info = env.step(last_obs['sampled_action'])
                buff.add(
                    last_obs,
                    obs,
                    last_obs['sampled_action'],
                    reward,
                    done,
                    info,
                )
                count += 1
                last_obs = obs
                total_reward += reward

        losses = []
        L1 = []
        L1_DEPTH = []
        SSIM_1 = []
        SSIM_2 = []
        SSIM_DEPTH = []

        # Model Update
        for _ in range(count):
            rollout = buff.sample(batch_size)
            scale_1 = rollout.observations['scale_1']
            scale_2 = rollout.observations['scale_2']
            gt_image = torch.cat([
                scale_1, scale_2
            ], 1).float() / 255

            gt_depth = rollout.observations['depth']

            # Prediction
            _, [gen_image, depth] = model(gt_image.contiguous())

            # Gradient Computatation and Optimsation
            l1_gen_image = torch.nn.functional.l1_loss(gen_image, gt_image)
            l1_depth = torch.nn.functional.l1_loss(depth, gt_depth)
            L1.append(l1_gen_image.item())
            L1_DEPTH.append(l1_depth.item())

            # SSIM computation
            ssim_scale_1 = 1 - ssim(
                gt_image[:, :3], gen_image[:, :3],
                data_range=1, size_average=True
            )
            ssim_scale_2 = 1 - ssim(
                gt_image[:, 3:], gen_image[:, 3:],
                data_range=1, size_average=True
            )
            ssim_depth = 1 - ssim(
                gt_depth, depth,
                data_range=1, size_average=True
            )
            SSIM_1.append(ssim_scale_1.item())
            SSIM_2.append(ssim_scale_2.item())
            SSIM_DEPTH.append(ssim_depth.item())
            loss = l1_depth + l1_gen_image + ssim_scale_1 + ssim_scale_2 + ssim_depth

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())

        
        # Logging
        writer.add_scalar('Train/Loss', np.mean(losses), i)
        writer.add_scalar('Train/L1', np.mean(L1), i)
        writer.add_scalar('Train/ssim_1', np.mean(SSIM_1), i)
        writer.add_scalar('Train/ssim_2', np.mean(SSIM_2), i)
        writer.add_scalar('Train/ssim_depth', np.mean(SSIM_DEPTH), i)
        writer.add_scalar('Train/depth', np.mean(L1_DEPTH), i)
        writer.add_scalar('Train/learning_rate', scheduler.get_last_lr()[0], i)
        print('Epoch {} Learning Rate {:.6f} Total Reward {:.4f} Loss {:.4f} L1 {:.4f} L1 depth {:.4f} SSIM_1 {:.4f} SSIM_2 {:.4f} SSIM_DEPTH {:.4f} steps {}'.format(
            i, scheduler.get_last_lr()[0], total_reward[0], np.mean(losses), np.mean(L1), np.mean(L1_DEPTH),
            np.mean(SSIM_1), np.mean(SSIM_2), np.mean(SSIM_DEPTH), count))

        # Save Model
        if i % save_freq == 0:
            state_dict = { 
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optim.state_dict(),
            }
            torch.save(state_dict, os.path.join(logdir, 'model_epoch_{}.pt'.format(i)))
        scheduler.step()
