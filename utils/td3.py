from typing import Any, Dict, List, Optional, Type, Union, Tuple

import stable_baselines3 as sb3
from bg.autoencoder import Autoencoder
import torch
import gym
import numpy as np
from pytorch_msssim import ssim, ms_ssim
import copy
from constants import params

class FeaturesExtractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self,
        observation_space: gym.Space,
        features_dim: int,
        pretrained_params_path = 'assets/out/models/autoencoder/model.pt',
        device = None,
    ):
        super(FeaturesExtractor, self).__init__(observation_space, features_dim)
        self.vc = Autoencoder(
            [1, 1, 1, 1],
            features_dim,
            3
        )

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
        self.vc.load_state_dict(
            torch.load(pretrained_params_path, map_location = torch.device(device))['model_state_dict']
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(512, features_dim),
            torch.nn.Tanh()
        )

        self.fc_sensors = torch.nn.Sequential(
            torch.nn.Linear(
                observation_space['sensors'].shape[-1],
                2 * observation_space['sensors'].shape[-1]
            ),
            torch.nn.Tanh()
        )

        self.combine = torch.nn.Sequential(
            torch.nn.Linear(features_dim + 2 * observation_space['sensors'].shape[-1], features_dim),
            torch.nn.Tanh()
        )

    def forward(self, observations):
        image = torch.cat([
            observations['scale_1'], observations['scale_2'], observations['scale_3']
        ], 1)
        visual, [gen_image, depth] = self.vc(image)
        visual = torch.nn.functional.adaptive_avg_pool2d(visual, 1)
        visual = visual.view(visual.size(0), -1)
        visual = self.linear(visual)

        sensors = self.fc_sensors(observations['sensors'])

        features = torch.cat([visual, sensors], -1)
        features = self.combine(features)
        return features, [gen_image, depth]

class Actor(sb3.common.policies.BasePolicy):
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
        activation_fn: Type[torch.nn.Module] = torch.nn.Tanh,
        normalize_images: bool = True,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = sb3.common.preprocessing.get_action_dim(self.action_space)
        squash_output = True
        actor_net = sb3.common.torch_layers.create_mlp(
            features_dim,
            action_dim,
            net_arch,
            activation_fn,
            squash_output = squash_output
        )

        if squash_output:
            torch.nn.init.uniform_(actor_net[-2].weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(actor_net[-2].bias, -3e-4, 3e-4)
        else:
            torch.nn.init.uniform_(actor_net[-1].weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(actor_net[-1].bias, -3e-4, 3e-4)

        # Deterministic action
        self.mu = torch.nn.Sequential(*actor_net)

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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features, [gen_image, depth] = self.extract_features(obs)
        return self.mu(features), [gen_image, depth]

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.forward(observation)

    def predict(self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            
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
            actions, [gen_image, depth] = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()
        gen_image = gen_image.cpu().numpy()
        depth = depth.cpu().numpy()

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

        return [actions, [gen_image, depth]], state

class ContinuousCritic(sb3.common.policies.BaseModel):
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
        activation_fn: Type[torch.nn.Module] = torch.nn.Tanh,
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
            q_net = sb3.common.torch_layers.create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = torch.nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features, _ = self.extract_features(obs)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features, _ = self.extract_features(obs)
        return self.q_networks[0](torch.cat([features, actions], dim=1))

class TD3Policy(sb3.common.policies.BasePolicy):
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
        super(TD3Policy, self).__init__(
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
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
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

    def make_actor(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def predict(self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
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
            actions, [gen_image, depth] = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()
        gen_image = gen_image.cpu().numpy()
        depth = depth.cpu().numpy()

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

        return [actions, [gen_image, depth]], state

class TD3(sb3.TD3):
    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
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
    ):
        super(TD3, self).__init__(
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
        if self.num_timesteps < params['staging_steps']:
            # Pretraining Phase
            unscaled_action = self._last_obs['sampled_action']
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            [unscaled_action, _], _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None and self.num_timesteps > params['staging_steps']:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        reconstruction_losses = []
        supervised_losses = []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions, _ = self.actor_target(replay_data.next_observations)
                next_actions = (next_actions + noise).clamp(-1, 1)

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

            action, [gen_image, depth] = self.actor(replay_data.observations)
            image = torch.cat([
                replay_data.observations['scale_1'],
                replay_data.observations['scale_2'],
                replay_data.observations['scale_3']
            ], 1).float() / 255

            reconstruction_loss = torch.nn.functional.l1_loss(
                gen_image, image
            ) + torch.nn.functional.l1_loss(
                depth, replay_data.observations['depth']
            ) + 1 - ssim(
                image[:, :3], gen_image[:, :3],
                data_range=1.0, size_average=True
            ) + 1 - ssim(
                    image[:, 3:6], gen_image[:, 3:6],
                data_range=1.0, size_average=True
            ) + 1 - ssim(
                image[:, 6:], gen_image[:, 6:],
                data_range=1.0, size_average=True
            )
            reconstruction_losses.append(reconstruction_loss.item())

            # Delayed policy updates
            # Compute actor loss
            if self.num_timesteps < params['staging_steps']:
                actor_loss = torch.nn.functional.mse_loss(action, replay_data.observations['scaled_sampled_action'])
                supervised_losses.append(actor_loss.item())
            else:
                actor_loss = -self.critic.q1_forward(replay_data.observations, action).mean()
                actor_losses.append(actor_loss.item())
            actor_loss += reconstruction_loss

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            sb3.common.utils.polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            sb3.common.utils.polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/reconstruction", np.mean(reconstruction_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))

        if len(supervised_losses) > 0:
            self.logger.record("train/supervised_loss", np.mean(supervised_losses))

class Imitate(sb3.TD3):
    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
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
    ):
        super(Imitate, self).__init__(
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
        unscaled_action = self._last_obs['sampled_action']

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        reconstruction_losses = []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions, _ = self.actor_target(replay_data.next_observations)
                next_actions = (next_actions + noise).clamp(-1, 1)

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

            action, [gen_image, depth] = self.actor(replay_data.observations)
            image = torch.cat([
                replay_data.observations['scale_1'],
                replay_data.observations['scale_2'],
                replay_data.observations['scale_3']
            ], 1).float() / 255

            reconstruction_loss = torch.nn.functional.l1_loss(
                gen_image, image
            ) + torch.nn.functional.l1_loss(
                depth, replay_data.observations['depth']
            ) + 1 - ssim(
                image[:, :3], gen_image[:, :3],
                data_range=1.0, size_average=True
            ) + 1 - ssim(
                    image[:, 3:6], gen_image[:, 3:6],
                data_range=1.0, size_average=True
            ) + 1 - ssim(
                image[:, 6:], gen_image[:, 6:],
                data_range=1.0, size_average=True
            )
            reconstruction_losses.append(reconstruction_loss.item())

            actor_loss = reconstruction_loss + torch.nn.functional.mse_loss(action, replay_data.observations['scaled_sampled_action'])
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            sb3.common.utils.polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            sb3.common.utils.polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/reconstruction", np.mean(reconstruction_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
