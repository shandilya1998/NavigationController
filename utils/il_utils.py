from typing import Any, Dict, List, Optional, Tuple, Type, Union, NamedTuple
import stable_baselines3 as sb3
import torch
import os
import numpy as np
import gym
from utils.td3_utils import TD3BGPolicy
from constants import params
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

TensorDict = Dict[Union[str, int], torch.Tensor]

class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

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
    ):
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

        # only 1 env is supported
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype = action_space.dtype)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype = np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype = np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.returns.nbytes + self.advantages.nbytes

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
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        for key in self.observations.keys():
            obs_ = obs[key].clone().cpu().numpy()
            self.observations[key][self.pos] = obs_

        self.actions[self.pos] = action.clone().cpu().numpy()
        self.returns[self.pos] = returns.clone().cpu().numpy()
        self.advantages[self.pos] = advantages.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[sb3.common.vec_env.VecNormalize] = None) ->  DictRolloutBufferSamples:
        """
        Sample elements from the replay buffer.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(sb3.common.buffers.ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> DictRolloutBufferSamples:

        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds, 0]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )

class ImitationLearning(sb3.common.on_policy_algorithm.OnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[TD3BGPolicy]],
        env: Union[sb3.common.type_aliases.GymEnv, str],
        learning_rate: Union[float, sb3.common.type_aliases.Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cuda",
        _init_setup_model: bool = True,
    ):
        super(ImitationLearning, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0.0,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                gym.spaces.Box,
                gym.spaces.Discrete,
                gym.spaces.MultiDiscrete,
                gym.spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = torch.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()
        
        self.replay_buffer = DictReplayBuffer(
            int(n_steps * 10),
            self.observation_space,
            self.action_space,
            self.device,
            n_envs = self.n_envs,
        )

    def collect_rollouts(
        self,
        env: sb3.common.vec_env.VecEnv,
        callback: sb3.common.callbacks.BaseCallback,
        rollout_buffer: sb3.common.buffers.RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = sb3.common.utils.obs_as_tensor(self._last_obs, self.device)
                actions, values, advantages = self.policy.forward(obs_tensor)
                actions = obs_tensor['sampled_action']
                actions = actions[:, :2]
            actions = actions.cpu().numpy()
            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(np.concatenate([clipped_actions, values.cpu().numpy()], -1))

            self.num_timesteps += env.num_envs
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, advantages)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            obs_tensor = sb3.common.utils.obs_as_tensor(new_obs, self.device)
            actions, values, advantages = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        for rollout in rollout_buffer.get(None):
            observations = rollout.observations
            actions = rollout.actions
            returns = torch.unsqueeze(rollout.returns, -1)
            advantages = torch.unsqueeze(rollout.advantages, -1)
            for i in range(n_rollout_steps):
                self.replay_buffer.add(
                    {key: observations[key][i] for key in observations.keys()},
                    actions[i],
                    returns[i],
                    advantages[i]
                )

        callback.on_rollout_end()

        return True


    def train(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        i = 0
        while i < params['n_updates']:
            replay_data = self.replay_buffer.sample(batch_size=params['batch_size'])
            # TODO: avoid second computation of everything because of the gradient
            actions, values, pred_advantages = self.policy.forward(replay_data.observations)
            # Policy gradient loss
            advantages = torch.unsqueeze(replay_data.advantages, -1)
            returns = torch.unsqueeze(replay_data.returns, -1)
            advantage_loss = torch.nn.functional.mse_loss(advantages, pred_advantages)
            if self.normalize_advantage:
                pred_advantages = (pred_advantages - pred_advantages.mean()) / pred_advantages.std()   
            policy_loss = -pred_advantages.mean()
            action_loss = torch.nn.functional.mse_loss(replay_data.actions, actions)
            # Value loss using the TD(gae_lambda) target
            value_loss = torch.nn.functional.mse_loss(returns, values)

            loss = policy_loss + action_loss + self.vf_coef * value_loss + self.vf_coef * advantage_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            i += 1
            self._n_updates += 1
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/policy_loss", policy_loss.item())
            self.logger.record("train/value_loss", value_loss.item())
            self.logger.record("train/advantage_loss", advantage_loss.item())
            self.logger.record("train/action_loss", action_loss.item())


class _ImitationLearning(sb3.TD3):
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
        policy: Union[str, Type[sb3.td3.policies.TD3Policy]],
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
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(_ImitationLearning, self).__init__(
            policy = policy,
            env = env,
            learning_rate = learning_rate,
            buffer_size = buffer_size,
            learning_starts = learning_starts,
            batch_size = batch_size,
            tau = tau,
            gamma = gamma,
            train_freq = train_freq,
            gradient_steps = gradient_steps,
            action_noise = action_noise,
            replay_buffer_class = replay_buffer_class,
            replay_buffer_kwargs = replay_buffer_kwargs,
            optimize_memory_usage = optimize_memory_usage,
            policy_delay = policy_delay,
            target_policy_noise = target_policy_noise,
            target_noise_clip = target_noise_clip,
            tensorboard_log = tensorboard_log,
            create_eval_env = create_eval_env,
            policy_kwargs = policy_kwargs,
            verbose = verbose,
            seed = seed,
            device = device,
            _init_setup_model = _init_setup_model
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum([torch.nn.functional.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                predicted_actions = self.actor(replay_data.observations)
                actor_loss = torch.nn.functional.mse_loss(
                    predicted_actions,
                    replay_data.actions
                ) 

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                sb3.common.utils.polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                sb3.common.utils.polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))    

    def _sample_action(
        self, learning_starts: int,
        action_noise: Optional[sb3.common.noise.ActionNoise] = None
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
        #unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        unscaled_action = self._last_obs['sampled_action']


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

