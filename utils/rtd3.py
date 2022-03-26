import numpy as np
import gym
import stable_baselines3 as sb3
from typing import NamedTuple, Any, Dict, List, Optional, Tuple, Type, Union
import torch
from constants import params


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
        super(
            EpisodicDictReplayBuffer,
            self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs)
        assert isinstance(
            self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        assert n_envs == 1, "Replay buffer only support single environment for now"
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros(
                (self.n_ep,
                 self.max_ep_size,
                 self.n_envs) + _obs_shape,
                dtype=observation_space[key].dtype) for key,
            _obs_shape in self.obs_shape.items()}
        self.next_observations = {
            key: np.zeros(
                (self.n_ep,
                 self.max_ep_size,
                 self.n_envs) + _obs_shape,
                dtype=observation_space[key].dtype) for key,
            _obs_shape in self.obs_shape.items()}

        # only 1 env is supported
        self.actions = np.zeros(
            (self.n_ep,
             self.max_ep_size,
             self.n_envs,
             self.action_dim),
            dtype=action_space.dtype)
        self.rewards = np.zeros(
            (self.n_ep, self.max_ep_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros(
            (self.n_ep,
             self.max_ep_size,
             self.n_envs),
            dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros(
            (self.n_ep, self.max_ep_size, self.n_envs), dtype=np.float32)
        self.episode_lengths = np.zeros(
            (self.n_ep, self.n_envs), dtype=np.int32)
        self.ep = 0
        self.pos = np.zeros((self.n_envs), dtype=np.int32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + \
                self.rewards.nbytes + self.dones.nbytes + self.episode_lengths.nbytes
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
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB")

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
            self.observations[key][self.ep,
                                   self.pos] = np.array(obs[key]).copy()

        for key in self.next_observations.keys():
            self.next_observations[key][self.ep,
                                        self.pos] = np.array(next_obs[key]).copy()

        self.actions[self.ep, self.pos] = np.array(action).copy()
        self.rewards[self.ep, self.pos] = np.array(reward).copy()
        self.dones[self.ep, self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.ep, self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos])

        for i, d in enumerate(done):
            if d or self.pos[i] >= self.max_ep_size - 1:
                if self.pos > self.max_ep_size * 0.1:
                    self.episode_lengths[self.ep, i] = self.pos[i] + 1
                    self.ep += 1
                self.pos[i] = 0
            else:
                self.pos[i] += 1
            if i >= 1:
                raise ValueError(
                    "Replay buffer only support single environment for now")
        if self.ep == self.n_ep:
            self.full = True
            self.ep = 0

    def __get_batch_inds(self, batch_size):
        if not self.full:
            batch_inds = np.random.randint(0, self.ep, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.n_ep, size=batch_size)
        return batch_inds

    def sample(self, batch_size: int,
               env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None):
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

        return self._get_samples(batch_size = batch_size, env=env)

    def _get_samples(self, batch_size: int,
                     env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None):
       
        batch_inds = self.__get_batch_inds(batch_size=batch_size)
        size = int(self.episode_lengths[batch_inds].min())
        count = 0
        index = 0
        """
            Need to ensure that the entire sequence will be seen by the neural networks with the same probability
        """
        while count < self.maz_ep_size:
            if index >= size:
                batch_inds = self.__get_batch_inds(batch_size=batch_size)
                index = 0
            obs = self._normalize_obs(
                {key: obs[batch_inds, index, 0, :] for key, obs in self.observations.items()})
            
            next_obs = self._normalize_obs(
                {key: obs[batch_inds, index, 0, :] for key, obs in self.next_observations.items()})

            observations = {
                key: self.to_torch(item) for key,
                item in obs.items()}
            next_observations = {
                key: self.to_torch(item) for key,
                item in next_obs.items()}
            actions = self.actions[batch_inds, index, 0, :]
            dones = self.dones[batch_inds, index] * \
                (1 - self.timeouts[batch_inds, index])
            rewards = self._normalize_reward(self.rewards[batch_inds, index], env)
            count += 1
            index += 1
            yield sb3.common.type_aliases. DictReplayBufferSamples(
                observations=observations,
                actions=self.to_torch(actions),
                next_observations=next_observations,
                dones=self.to_torch(dones),
                rewards=self.to_torch(rewards),
                size=size
            )
