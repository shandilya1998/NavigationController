import stable_baselines3 as sb3
import gym
from constants import params
from typing import Any, Dict
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Type, Union, NamedTuple
from bg.models import ControlNetwork
import warnings
from abc import ABC, abstractmethod
import inspect

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class SaveOnBestTrainingRewardCallback(sb3.common.callbacks.BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = sb3.common.results_plotter.ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
        return True

class CustomCallback(sb3.common.callbacks.BaseCallback):
    def __init__(self,
        eval_env: gym.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(
                _locals: Dict[str, Any],
                _globals: Dict[str, Any]
            ) -> None:
                """
                Renders the environment in its current state,
                    recording the screen in the captured `screens` list

                :param _locals:
                    A dictionary containing all local variables of the callback's scope
                :param _globals:
                    A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))
            
            sb3.common.evaluation.evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                sb3.common.logger.Video(torch.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
            for item in params['track_list']:
                ITEM = np.stack(self._eval_env.env.env._track_item[item], 0)
                fig, ax = plt.subplots(
                    1, ITEM.shape[-1],
                    figsize = (7.5 * ITEM.shape[-1], 7.5)
                )
                T = np.arange(ITEM.shape[0]) * self._eval_env.dt
                if ITEM.shape[-1] > 1:
                    for j in range(ITEM.shape[-1]):
                        ax[j].plot(T, ITEM[:, j], color = 'r',
                            label = '{}_{}'.format(item, j),
                            linestyle = '--')
                        ax[j].set_xlabel('time', fontsize = 12)
                        ax[j].set_ylabel('{}'.format(item), fontsize = 12)
                        ax[j].legend(loc = 'upper left')
                else:
                    ax.plot(T, ITEM[:, 0], color = 'r',
                        label = '{}_{}'.format(item, 0), 
                        linestyle = '--')
                    ax.set_xlabel('time', fontsize = 12) 
                    ax.set_ylabel('{}'.format(item), fontsize = 12) 
                    ax.legend(loc = 'upper left')
                self.logger.record("trajectory/{}".format(item), 
                    sb3.common.logger.Figure(
                        fig, close=True
                    ),
                    exclude = ("stdout", "log", "json", "csv")
                )
                plt.close()
        return True


class PassAsIsFeaturesExtractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super(PassAsIsFeaturesExtractor, self).__init__(observation_space, sb3.common.preprocessing.get_flattened_obs_dim(observation_space))

    def forward(self, observations):
        return [observations['observation'], observations['state_value']]

class ActorBG(sb3.common.policies.BasePolicy):
    """
    Actor network (policy) for TD3BG.

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
        net_params: Dict[str, Any],
        features_extractor: torch.nn.Module,
        features_dim: int,
        normalize_images: bool = True,
    ):
        super(ActorBG, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # 2 is being subtracted to account for the additional value and advantage values in the policy output
        action_dim = sb3.common.preprocessing.get_action_dim(self.action_space) - 2
        if action_dim <= 0:
            raise ValueError('Action Space Must contain atleast 3 dimensions, got 2 or less')
        """
            Requires addition of the basal ganglia model here in the next two
            statements
        """
        # Deterministic action
        self.net_params = net_params
        net_params['action_dim'] = action_dim
        self.mu = ControlNetwork(**net_params)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_params=self.net_params,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        return self.mu(features)

    def _predict(self, observation: List[torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor]:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.forward(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

class TD3BGPolicy(sb3.common.policies.BasePolicy):
    """
    Policy class (with both actor and critic in the same network) for TD3BG.

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
        net_params: Optional[Dict[str, int]] = {},
        features_extractor_class: Type[sb3.common.torch_layers.BaseFeaturesExtractor] = PassAsIsFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
    ):
        super(TD3BGPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        self.net_params = net_params
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_params": self.net_params,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        self.actor, self.actor_target = None, None
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

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_params=self.net_params,
                lr_schedule=self._dummy_schedule,  
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> ActorBG:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ActorBG(**actor_kwargs).to(self.device)

    def forward(self, observation: List[torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor]:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: List[torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor]:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)
        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        low = low[:-2]
        high = high[:-2]
        ac = action[:, :-2]
        values = action[:, -2:]
        scaled_ac = 2.0 * ((ac - low) / (high - low)) - 1.0
        return np.concatenate([scaled_ac, values], -1)

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        low = low[:-2]
        high = high[:-2]
        ac = scaled_action[:, :-2]
        values = scaled_action[:, -2:]
        unscaled_ac = low + (0.5 * (ac + 1.0) * (high - low))
        return np.concatenate([unscaled_ac, values], -1)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.training = mode

sb3.common.policies.register_policy('MlpBGPolicy', TD3BGPolicy)

class DictReplayBufferBG(sb3.common.buffers.DictReplayBuffer):
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
        n_steps: int = 2,
    ):
        self.n_steps = n_steps
        super(DictReplayBufferBG, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        assert self.n_steps > 1, "n_steps must be greater than 1 to have multiple steps"
        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size + self.n_steps, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        # only 1 env is supported
        self.actions = np.zeros((self.buffer_size + self.n_steps, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size + self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size + self.n_steps, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size + self.n_steps, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
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
            self.observations[key][self.pos] = np.array(obs[key]).copy()
            self.observations[key][(self.pos + 1) % (self.buffer_size + self.n_steps)] = np.array(next_obs[key]).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size + self.n_steps:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> sb3.common.type_aliases.DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> sb3.common.type_aliases.DictReplayBufferSamples:
        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = {key: [] for key in self.observations.keys()}
        next_obs_ = {key: [] for key in self.observations.keys()}
        actions = []
        dones = []
        rewards = []
        for i, index in enumerate(batch_inds):
            ob = self._normalize_obs({key: obs[index:index + 1, 0, :] for key, obs in self.observations.items()})
            next_ob = self._normalize_obs({key: obs[index + 1:index + 1 + self.n_steps, 0, :] for key, obs in self.observations.items()})
            for key in self.observations.keys():
                obs_[key].append(ob[key])
                next_obs_[key].append(next_ob[key])
            actions.append(self.actions[index: index + self.n_steps + 1])
            dones.append(
                self.dones[index: index + self.n_steps + 1] * (1 - self.timeouts[index: index + self.n_steps + 1])
            )
            rewards.append(self._normalize_reward(self.rewards[index: index + self.n_steps + 1], env))
        obs_ = self._normalize_obs({key: self.to_torch(np.stack(ob, 0)) for key, ob in obs_.items()})
        next_obs_ = self._normalize_obs({key: self.to_torch(np.stack(ob, 0)) for key, ob in next_obs_.items()})
        # Convert to torch tensor
        actions = self.to_torch(np.stack(actions, 0))
        dones=self.to_torch(np.stack(dones, 0))
        rewards=self.to_torch(np.stack(rewards, 0))
        return sb3.common.type_aliases.DictReplayBufferSamples(
            observations = obs_,
            actions = actions,
            next_observations = next_obs_,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones = dones,
            rewards = rewards,
        )

class TD3BG(sb3.common.off_policy_algorithm.OffPolicyAlgorithm):
    """
    Twin Delayed DDPG for Basal Ganglia (TD3BG)
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
        policy: Union[str, Type[TD3BGPolicy]],
        env: Union[sb3.common.type_aliases.GymEnv, str],
        learning_rate: Union[float, sb3.common.type_aliases.Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        n_steps: int = 2,
        tau: float = 0.005,
        lmbda: float = 0.01,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[sb3.common.noise.ActionNoise] = None,
        replay_buffer_class: DictReplayBufferBG = DictReplayBufferBG,
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
        self.lmbda = lmbda
        self.n_steps = n_steps
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {'n_steps' : n_steps}
        else:
            replay_buffer_kwargs['n_steps'] = n_steps
        super(TD3BG, self).__init__(
            policy,
            env,
            TD3BGPolicy,
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
        super(TD3BG, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                current_value = replay_data.actions[:, 0, -2:-1].clone()
                target_q_values = torch.zeros(current_value.size())
                target_advantage_values = torch.zeros(current_value.size())
                status = (1 - replay_data.dones[:, 0, :].clone())
                for i in range(1, self.n_steps + 1):
                    status = (1 - replay_data.dones[:, i, :].clone()) * status
                    next_actions = self.actor_target({
                        key: replay_data.next_observations[key][:, i - 1, :] for key in replay_data.next_observations.keys()
                    })
                    next_value = next_actions[:, -2:-1].clone()
                    next_advantage = next_actions[:, -1:].clone()
                    next_q_values = next_value + next_advantage
                    value = replay_data.actions[:, i, -2:-1]
                    sum_reward = torch.zeros(replay_data.rewards[:, i, :].size())
                    for j in range(i):
                        sum_reward += replay_data.rewards[:, j, :] * status * (self.gamma ** j)
                    target_advantage_values += (-current_value + sum_reward + next_value * (self.gamma ** i)) * (self.lmbda ** (i - 1))
                    target_q_values += sum_reward + status * (self.gamma ** i) * next_q_values
                target_q_values = target_q_values / self.n_steps
                target_advantage_values = (1 - self.lmbda) * target_advantage_values

            # Get current Q-values estimates for each critic network
            actions = self.actor({
                key: replay_data.observations[key][:, 0, :] for key in replay_data.observations.keys()
            })
            current_advantage_values = actions[:, -1:]
            current_q_values = actions[:, -2:-1] + current_advantage_values

            # Compute critic loss
            critic_loss = torch.nn.functional.mse_loss(current_q_values, target_q_values) + \
                torch.nn.functional.mse_loss(current_advantage_values, target_advantage_values)
            critic_losses.append(critic_loss.item())

            loss = critic_loss
            # Optimize the critics
            self.actor.optimizer.zero_grad()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                ac = self.actor({
                    key: replay_data.observations[key][:, 0, :] for key in replay_data.observations.keys()
                })
                q = ac[:, -2:-1] + ac[:, -1:]
                actor_loss = -q.mean()
                actor_losses.append(actor_loss.item())

                loss += actor_loss
                # Optimize the actor
            loss.backward()
            self.actor.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                sb3.common.utils.polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

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
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_ac = np.clip(scaled_action[:, :-2] + action_noise(), -1, 1)
                scaled_action = np.concatenate([scaled_ac, scaled_action[:, -2:]], -1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

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

        return super(TD3BG, self).learn(
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
        return super(TD3BG, self)._excluded_save_params() + ["actor", "actor_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer",]
        return state_dicts, []
