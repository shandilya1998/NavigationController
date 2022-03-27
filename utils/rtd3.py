import warnings
import numpy as np
import gym
import stable_baselines3 as sb3
from typing import NamedTuple, Any, Dict, List, Optional, Tuple, Union
import torch
import psutil

"""
Idea of burn in comes from the following paper:
https://openreview.net/pdf?id=r1lyTjAqYX
"""

TensorDict = Dict[Union[str, int], torch.Tensor]
TensorList = List[torch.Tensor]


class ReccurentDictReplayBufferSamples(NamedTuple):
    prev_observations: TensorDict
    prev_states: TensorList
    observations: TensorDict
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    states: TensorList


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
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        state_spec: Optional[List[Tuple[Tuple[int], torch.dtype]]] = None,
        max_seq_len: int = 30,
        burn_in_seq_len: int = 10
    ):
        self.max_seq_len = max_seq_len
        self.burn_in_seq_len = burn_in_seq_len
        super(sb3.common.buffers.ReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

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
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype) for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype) for key, _obs_shape in self.obs_shape.items()
        }

        assert state_spec is not None

        self.states = [
            np.zeros((self.buffer_size, self.n_envs) + s, dtype=d)
            for s, d in state_spec
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
        states: List[np.ndarray],
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            self.observations[key][self.pos] = np.array(obs[key]).copy()

        for key in self.next_observations.keys():
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        for i, state in enumerate(states):
            self.states[i][self.pos] = np.array(state).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> ReccurentDictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(sb3.common.buffers.ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def __expand_include(self, include, shape):
        length = len(include.shape)
        assert shape[:length] == include.shape
        shape = shape[length:]
        for item in shape:
            include = np.repeat(np.expand_dims(include, -1), item, -1)
        return include

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> ReccurentDictReplayBufferSamples:
        
        # Collecting previous observations and states for "burn in" state computation
        offsets = np.repeat(np.expand_dims(np.arange(-self.burn_in_seq_len, 0), 0), len(batch_inds), 0)
        prev_inds = np.repeat(np.expand_dims(batch_inds, 1), self.burn_in_seq_len, 1) + offsets
        offset = self.buffer_size if self.full else self.pos
        prev_inds[prev_inds < 0] = prev_inds[prev_inds < 0] + offset
        prev_done_inds = prev_inds
        prev_dones = self.dones[prev_done_inds, 0]
        include = np.flip(np.multiply.accumulate(np.flip(1 - prev_dones, 1), 1), 1)
        prev_obs = self._normalize_obs({
            key: obs[prev_inds, 0, :] * self.__expand_include(include, obs[prev_inds, 0, :].shape).astype(obs.dtype) for key, obs in self.observations.items()
            })
        prev_states = [state[prev_inds[:, 0], 0] * np.prod(include, 1) for state in self.states]

        # Computing indices for timesteps used for gradient computation
        inds = np.repeat(np.expand_dims(batch_inds, 1), self.max_seq_len, 1)
        offsets = np.repeat(np.expand_dims(np.arange(0, self.max_seq_len), 0), len(batch_inds))
        inds = inds + offsets
        max_ind = self.buffer_size - 1 if self.full else self.pos
        inds[inds > max_ind] = inds[inds > max_ind] - max_ind
        dones = self.dones[inds]
        size = self.max_seq_len
        all_dones = np.where(dones == 1)[1]
        if len(all_dones) > 0:
            size = all_dones.min()
        inds = inds[:, :size]

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        dones = dones[:, :size] * (1 - self.timeouts[inds])
        obs = self._normalize_obs({key: obs[inds, 0, :] for key, obs in self.observations.items()})
        next_obs = self._normalize_obs({key: obs[inds, 0, :] for key, obs in self.next_observations.items()})
        actions = self.actions[inds]
        rewards = self._normalize_reward(self.rewards[inds], env)
        states = [
                state[inds] for state in self.states
                ]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}
        prev_observations = {key: self.to_torch(obs) for key, obs in prev_obs.items()}
        prev_states = [self.to_torch(state) for state in prev_states]
        actions = self.to_torch(actions)
        rewards = self.to_torch(rewards)
        dones = self.to_torch(dones)
        states = [self.to_torch(state) for state in states]

        return ReccurentDictReplayBufferSamples(
            prev_observations=prev_observations,
            prev_states=prev_states,
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            states=states
        )
