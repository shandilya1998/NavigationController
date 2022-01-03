import torch
import numpy as np
import collections
import warnings
from functools import partial
import gym
from typing import Any, Dict, List, Optional, Tuple, Type, Union, NamedTuple
import stable_baselines3 as sb3
from bg.models import ControlNetwork, VisualCortex, ControlNetworkV2, \
    VisualCortexV2
import copy
from constants import params
import time
from collections import deque
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class PassAsIsFeaturesExtractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super(PassAsIsFeaturesExtractor, self).__init__(observation_space, params['num_ctx'])
        self.vc = VisualCortex(
            observation_space,
            params['num_ctx']
        )

    def forward(self, observations):
        ob_t = self.vc(observations['observation'])
        ob_t_1 = self.vc(observations['last_observation'])
        return [
            ob_t,
            ob_t_1
        ]

class MultiModalHistoryFeaturesExtractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, n_steps):
        features_dim = params['num_ctx']
        super(MultiModalHistoryFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.vc = VisualCortexV2(
            observation_space['observation'],
            params['num_ctx']
        )
        self.keys = ['observation', 'inertia', 'action']
        input_size = len(self.keys) * features_dim
        self.input_size = input_size
        self.fc_inertia = torch.nn.Sequential(
            torch.nn.Linear(observation_space['inertia'].shape[-1], features_dim),
            torch.nn.ReLU()
        )
        self.fc_history = torch.nn.Sequential(
            torch.nn.Linear(observation_space['action'].shape[-1], features_dim),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, features_dim),
            torch.nn.ReLU(),
        )
        self.layers = {}

    def forward(self, observations):
        out = []
        out.append(self.fc_inertia(observations['inertia']))
        out.append(self.fc_history(observations['action']))
        out.append(self.vc(observations['observation']))
        out = torch.cat(out, -1)
        return self.fc(out)

class MultiModalFeaturesExtractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        features_dim = params['num_ctx']
        super(MultiModalFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.vc = VisualCortex(
            observation_space,
            features_dim
        )
        input_size = (len(observation_space) - 1) * features_dim
        self.fc_inertia = torch.nn.Sequential(
            torch.nn.Linear(observation_space['inertia'].shape[-1], features_dim),
            torch.nn.ReLU()
        )
        self.fc_history = torch.nn.Sequential(
            torch.nn.Linear(observation_space['history'].shape[-1], features_dim),
            torch.nn.ReLU()
        ) 
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, features_dim),
            torch.nn.ReLU()
        )

    def forward(self, observations):
        vision = self.vc(observations['observation'])
        inertia = self.fc_inertia(observations['inertia'])
        history = self.fc_history(observations['history'])
        x = torch.cat([vision, inertia, history], -1)
        out = self.fc(x)
        return out

class MultiModalFeaturesExtractorV2(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        features_dim = params['num_ctx']
        super(MultiModalFeaturesExtractorV2, self).__init__(observation_space, features_dim)

    def forward(self, observations):
        return observations['observation'], observations['sensors']

class PassAsIsFeaturesExtractorV2(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super(PassAsIsFeaturesExtractorV2, self).__init__(observation_space, params['num_ctx'])
        self.vc = VisualCortex(
            observation_space,
            params['num_ctx']
        )    


    def forward(self, observations):
        ob_t = self.vc(observations['observation'])
        return ob_t

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
        action_dim = sb3.common.preprocessing.get_action_dim(self.action_space)
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
        action, vt, bg_out = self.forward(observation)
        return action

class SimpleVFActor(sb3.common.policies.BasePolicy):
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
        super(SimpleVFActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )   

        # 2 is being subtracted to account for the additional value and advantage values in the policy output
        action_dim = sb3.common.preprocessing.get_action_dim(self.action_space)
        if action_dim <= 0:
            raise ValueError('Action Space Must contain atleast 3 dimensions, got 2 or less')
        """ 
            Requires addition of the basal ganglia model here in the next two
            statements
        """
        # Deterministic action
        self.net_params = net_params
        net_params['action_dim'] = action_dim
        self.mu = ControlNetworkV2(**net_params)

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
        action, vt, bg_out = self.forward(observation)
        return action

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
            q_net = sb3.common.torch_layers.create_mlp(2 * features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = torch.nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
            features = torch.cat(features, -1)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs)
            features = torch.cat(features, -1)
        return self.q_networks[0](torch.cat([features, actions], dim=1))

class ContinuousCriticV2(sb3.common.policies.BaseModel):
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
            features = self.extract_features(obs)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](torch.cat([features, actions], dim=1))

class TD3BGPolicyV2(sb3.common.policies.BasePolicy):
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
        n_critics: int = 2,
        share_features_extractor: bool = True,
        use_sde: bool = False
    ):  
        super(TD3BGPolicyV2, self).__init__(
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
        self.critic_kwargs = self.net_args.copy()
        critic_arch = [400, 300]
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )
        del self.critic_kwargs['net_params']

        self.actor, self.actor_target =  None, None
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

        # Target networks should always be in eval mode

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_params=self.net_params,
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
    
    def make_actor(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> ActorBG:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return SimpleVFActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCriticV2(**critic_kwargs).to(self.device)

    def forward(self, observation: List[torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor]:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: List[torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor]:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        action, vt, bg_out = self.actor(observation)
        return action


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
        n_critics: int = 2, 
        share_features_extractor: bool = True,
        use_sde: bool = False
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
        self.critic_kwargs = self.net_args.copy()
        critic_arch = [400, 300]
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )
        del self.critic_kwargs['net_params']

        self.actor, self.actor_target =  None, None
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

        # Target networks should always be in eval mode

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_params=self.net_params,
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

    def make_actor(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> ActorBG:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ActorBG(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[sb3.common.torch_layers.BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: List[torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor]:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: List[torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor]:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        action, vt, bg_out = self.actor(observation)
        return action

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
        optimize_memory_usage: bool = True,
        handle_timeout_termination: bool = True,
    ):
        super(sb3.common.buffers.ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        #assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = { 
                key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
                for key, _obs_shape in self.obs_shape.items()
            }

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
            else:
                total_memory_usage /= 1e9
                mem_available /= 1e9
                print("Sufficient Memory Available. Using {} GB with {} GB available".format(total_memory_usage, mem_available))

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
        if self.optimize_memory_usage:
            for key in self.observations.keys():
                self.observations[key][(self.pos + 1) % self.buffer_size] = np.array(next_obs[key]).copy()
        else:
            for key in self.next_observations.keys():
                self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
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
            super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.VecNormalize] = None) -> sb3.common.type_aliases.DictReplayBufferSamples:
        obs_ = self._normalize_obs({key: obs[batch_inds, 0, :] for key, obs in self.observations.items()})
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        
        if self.optimize_memory_usage:
            next_obs_ = self._normalize_obs(
                {key: obs[
                    (batch_inds + 1) % self.buffer_size, 0, :
                ] for key, obs in self.observations.items()}
            )
        else:
            next_obs_ = self._normalize_obs({key: obs[batch_inds, 0, :] for key, obs in self.next_observations.items()})

        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return sb3.common.type_aliases.DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds] * (1 - self.timeouts[batch_inds])),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds], env)),
        )

class NStepReplayBuffer(sb3.common.buffers.ReplayBuffer):
    """
    Replay Buffer that computes N-step returns.
    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[torch.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    :param optimize_memory_usage: (bool) Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param n_steps: (int) The number of transitions to consider when computing n-step returns
    :param gamma:  (float) The discount factor for future rewards.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)
        self.n_steps = int(n_steps)
        if not 0 < n_steps <= buffer_size:
            raise ValueError("n_steps needs to be strictly smaller than buffer_size, and strictly larger than 0")
        self.gamma = gamma

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> sb3.common.type_aliases.ReplayBufferSamples:

        actions = self.actions[batch_inds, 0, :]

        gamma = self.gamma

        # Broadcasting turns 1dim arange matrix to 2 dimensional matrix that contains all
        # the indices, % buffersize keeps us in buffer range
        # indices is a [B x self.n_step ] matrix
        indices = (np.arange(self.n_steps) + batch_inds.reshape(-1, 1)) % self.buffer_size

        # two dim matrix of not dones. If done is true, then subsequent dones are turned to 0
        # using accumulate. This ensures that we don't use invalid transitions
        # not_dones is a [B x n_step] matrix
        not_dones = np.squeeze(1 - self.dones[indices], axis=-1)
        not_dones = np.multiply.accumulate(not_dones, axis=1)
        # vector of the discount factors
        # [n_step] vector
        gammas = gamma ** np.arange(self.n_steps)

        # two dim matrix of rewards for the indices
        # using indices we select the current transition, plus the next n_step ones
        rewards = np.squeeze(self.rewards[indices], axis=-1)
        rewards = self._normalize_reward(rewards, env)

        # TODO(PartiallyTyped): augment the n-step return with entropy term if needed
        # the entropy term is not present in the first step

        # if self.n_steps > 1: # not necessary since we assert 0 < n_steps <= buffer_size

        # # Avoid computing entropy twice for the same observation
        # unique_indices = np.array(list(set(indices[:, 1:].flatten())))

        # # Compute entropy term
        # # TODO: convert to pytorch tensor on the correct device
        # _, log_prob = actor.action_log_prob(observations[unique_indices, :])

        # # Memory inneficient version but fast computation
        # # TODO: only allocate the memory for that array once
        # log_probs = np.zeros((self.buffer_size,))
        # log_probs[unique_indices] = log_prob.flatten()
        # # Add entropy term, only for n-step > 1
        # rewards[:, 1:] = rewards[:, 1:] - ent_coef * log_probs[indices[:, 1:]]

        # we filter through the indices.
        # The immediate indice, i.e. col 0 needs to be 1, so we ensure that it is here using np.ones
        # If the jth transition is terminal, we need to ignore the j+1 but keep the reward of the jth
        # we do this by "shifting" the not_dones one step to the right
        # so a terminal transition has a 1, and the next has a 0
        filt = np.hstack([np.ones((not_dones.shape[0], 1)), not_dones[:, :-1]])

        # We ignore self.pos indice since it points to older transitions.
        # we then accumulate to prevent continuing to the wrong transitions.
        current_episode = np.multiply.accumulate(indices != self.pos, 1)

        # combine the filters
        filt = filt * current_episode

        # discount the rewards
        rewards = (rewards * filt) @ gammas
        rewards = rewards.reshape(len(batch_inds), 1).astype(np.float32)

        # Increments counts how many transitions we need to skip
        # filt always sums up to 1 + k non terminal transitions due to hstack above
        # so we subtract 1.
        increments = np.sum(filt, axis=1).astype(np.int) - 1

        next_obs_indices = (increments + batch_inds) % self.buffer_size
        obs = self._normalize_obs(self.observations[batch_inds, 0, :], env)
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(next_obs_indices + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[next_obs_indices, 0, :], env)

        dones = 1.0 - (not_dones[np.arange(len(batch_inds)), increments]).reshape(len(batch_inds), 1)

        data = (obs, actions, next_obs, dones, rewards)
        return sb3.common.type_aliases.ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class NStepHistoryReplayBuffer(sb3.common.buffers.ReplayBuffer):
    """
    Replay Buffer that computes N-step returns.
    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[torch.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    :param optimize_memory_usage: (bool) Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param n_steps: (int) The number of transitions to consider when computing n-step returns
    :param gamma:  (float) The discount factor for future rewards.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)
        self.n_steps = int(n_steps)
        if not 0 < n_steps <= buffer_size:
            raise ValueError("n_steps needs to be strictly smaller than buffer_size, and strictly larger than 0")

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> sb3.common.type_aliases.ReplayBufferSamples:
        actions = self.actions[batch_inds]
        # Broadcasting turns 1dim arange matrix to 2 dimensional matrix that contains all
        # the indices, % buffersize keeps us in buffer range
        # indices is a [B x self.n_step ] matrix
        indices = (-np.arange(self.n_steps) + batch_inds.reshape(-1, 1)) % self.buffer_size

        # two dim matrix of not dones. If done is true, then subsequent dones are turned to 0
        # using accumulate. This ensures that we don't use invalid transitions
        # not_dones is a [B x n_step] matrix
        dones = self.dones[batch_inds]
        not_dones = (1 - self.dones[indices])
        not_dones = np.multiply.accumulate(not_dones, axis=1)
        rewards = self._normalize_reward(self.rewards[batch_inds], env)

        include = not_dones.copy()
        for i in self.observation_shape.shape:
            include = np.repeat(np.expand_dims(include, -1), -1, i)
        obs = self._normalize_obs(self.observations[indices, 0, :], env) * include
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(indices + 1) % self.buffer_size, 0, :], env) * include
        else:
            next_obs = self._normalize_obs(self.next_observations[indices, 0, :], env) * include

        data = (obs, actions, next_obs, dones, rewards)
        return sb3.common.type_aliases.ReplayBufferSamples(*tuple(map(self.to_torch, data)))

class NStepHistoryDictReplayBuffer(sb3.common.buffers.DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        n_steps: int = 1
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)
        self.n_steps = int(n_steps)
        if not 0 < n_steps <= buffer_size:
            raise ValueError("n_steps needs to be strictly smaller than buffer_size, and strictly larger than 0")

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> sb3.common.type_aliases.ReplayBufferSamples:
        actions = self.actions[batch_inds]
        # Broadcasting turns 1dim arange matrix to 2 dimensional matrix that contains all
        # the indices, % buffersize keeps us in buffer range
        # indices is a [B x self.n_step ] matrix
        indices = (-np.arange(self.n_steps) + batch_inds.reshape(-1, 1)) % self.buffer_size

        # two dim matrix of not dones. If done is true, then subsequent dones are turned to 0
        # using accumulate. This ensures that we don't use invalid transitions
        # not_dones is a [B x n_step] matrix
        dones = self.dones[batch_inds]
        not_dones = (1 - self.dones[indices])
        not_dones = np.multiply.accumulate(not_dones, axis=1)
        rewards = self._normalize_reward(self.rewards[batch_inds], env) 

        obs = {} 
        next_obs = {} 
        for key, ob in self.observation_shape.keys():
            include = not_dones.copy()
            for i in self.observation_shape.shape:
                include = np.repeat(np.expand_dims(include, -1), -1, i)
            obs[key] = self._normalize_obs(self.observations[key][indices, 0, :], env) * include
            next_obs[key] = self._normalize_obs(self.next_observations[key][indices, 0, :], env) * include
    
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return sb3.common.type_aliases.DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(actions),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(dones),
            rewards=self.to_torch(self._normalize_reward(rewards)),
        )
        
class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor

class NStepLambdaReplayBuffer(sb3.common.buffers.ReplayBuffer):
    """
    Replay Buffer that computes N-step returns.
    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[torch.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    :param optimize_memory_usage: (bool) Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param n_steps: (int) The number of transitions to consider when computing n-step returns
    :param gamma:  (float) The discount factor for future rewards.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        lmbda: float = 0.9
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)
        self.n_steps = int(n_steps)
        if not 0 < n_steps <= buffer_size:
            raise ValueError("n_steps needs to be strictly smaller than buffer_size, and strictly larger than 0")
        self.gamma = gamma
        self.lmbda = lmbda

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> ReplayBufferSamples:
        gamma = self.gamma
        lmbda = self.lmbda
        # Broadcasting turns 1dim arange matrix to 2 dimensional matrix that contains all
        # the indices, % buffersize keeps us in buffer range
        # indices is a [B x self.n_step ] matrix
        indices = (np.arange(self.n_steps) + batch_inds.reshape(-1, 1)) % self.buffer_size

        # two dim matrix of not dones. If done is true, then subsequent dones are turned to 0
        # using accumulate. This ensures that we don't use invalid transitions
        # not_dones is a [B x n_step] matrix
        not_dones = np.squeeze(1 - self.dones[indices], axis=-1)
        not_dones = np.multiply.accumulate(not_dones, axis=1)
        dones = 1 - not_dones
        # vector of the discount factor
        # [n_step] vector
        gammas = gamma ** np.arange(self.n_steps)
        lmbdas = lmbda ** np.arange(self.n_steps)
        # two dim matrix of rewards for the indices
        # using indices we select the current transition, plus the next n_step ones
        rewards = np.squeeze(self.rewards[indices], axis=-1)
        rewards = self._normalize_reward(rewards, env) * not_dones
        returns = np.expand_dims(
            np.sum(lmbdas * np.add.accumulate(gammas * rewards), -1) * (1 - lmbda), -1
        ).astype(np.float32)

        obs = self._normalize_obs(self.observations[batch_inds, 0, :], env)
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(indices + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[indices, 0, :], env)

        actions = self.actions[batch_inds]
        data = (obs, actions, next_obs, dones, returns)
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

class NStepDictReplayBuffer(sb3.common.buffers.DictReplayBuffer):
    """
    Replay Buffer that computes N-step returns.
    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[torch.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    :param optimize_memory_usage: (bool) Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param n_steps: (int) The number of transitions to consider when computing n-step returns
    :param gamma:  (float) The discount factor for future rewards.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)
        self.n_steps = int(n_steps)
        if not 0 < n_steps <= buffer_size:
            raise ValueError("n_steps needs to be strictly smaller than buffer_size, and strictly larger than 0")
        self.gamma = gamma


    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> sb3.common.type_aliases.DictReplayBufferSamples:
        actions = self.actions[batch_inds]

        gamma = self.gamma

        # Broadcasting turns 1dim arange matrix to 2 dimensional matrix that contains all
        # the indices, % buffersize keeps us in buffer range
        # indices is a [B x self.n_step ] matrix
        indices = (np.arange(self.n_steps) + batch_inds.reshape(-1, 1)) % self.buffer_size

        # two dim matrix of not dones. If done is true, then subsequent dones are turned to 0
        # using accumulate. This ensures that we don't use invalid transitions
        # not_dones is a [B x n_step] matrix
        not_dones = np.squeeze(1 - self.dones[indices], axis=-1)
        not_dones = np.multiply.accumulate(not_dones, axis=1)
        # vector of the discount factors
        # [n_step] vector
        gammas = gamma ** np.arange(self.n_steps)

        # two dim matrix of rewards for the indices
        # using indices we select the current transition, plus the next n_step ones
        rewards = np.squeeze(self.rewards[indices], axis=-1)
        rewards = self._normalize_reward(rewards, env)

        # TODO(PartiallyTyped): augment the n-step return with entropy term if needed
        # the entropy term is not present in the first step

        # if self.n_steps > 1: # not necessary since we assert 0 < n_steps <= buffer_size

        # # Avoid computing entropy twice for the same observation
        # unique_indices = np.array(list(set(indices[:, 1:].flatten())))

        # # Compute entropy term
        # # TODO: convert to pytorch tensor on the correct device
        # _, log_prob = actor.action_log_prob(observations[unique_indices, :])

        # # Memory inneficient version but fast computation
        # # TODO: only allocate the memory for that array once
        # log_probs = np.zeros((self.buffer_size,))
        # log_probs[unique_indices] = log_prob.flatten()
        # # Add entropy term, only for n-step > 1
        # rewards[:, 1:] = rewards[:, 1:] - ent_coef * log_probs[indices[:, 1:]]

        # we filter through the indices.
        # The immediate indice, i.e. col 0 needs to be 1, so we ensure that it is here using np.ones
        # If the jth transition is terminal, we need to ignore the j+1 but keep the reward of the jth
        # we do this by "shifting" the not_dones one step to the right
        # so a terminal transition has a 1, and the next has a 0
        filt = np.hstack([np.ones((not_dones.shape[0], 1)), not_dones[:, :-1]])

        # We ignore self.pos indice since it points to older transitions.
        # we then accumulate to prevent continuing to the wrong transitions.
        current_episode = np.multiply.accumulate(indices != self.pos, 1)

        # combine the filters
        filt = filt * current_episode

        # discount the rewards
        rewards = (rewards * filt) @ gammas
        rewards = rewards.reshape(len(batch_inds), 1).astype(np.float32)

        # Increments counts how many transitions we need to skip
        # filt always sums up to 1 + k non terminal transitions due to hstack above
        # so we subtract 1.
        increments = np.sum(filt, axis=1).astype(np.int) - 1

        next_obs_indices = (increments + batch_inds) % self.buffer_size
        obs_ = self._normalize_obs({key: obs[batch_inds, 0, :] for key, obs in self.observations.items()})
        next_obs_ = self._normalize_obs({key: obs[next_obs_indices, 0, :] for key, obs in self.next_observations.items()})

        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        dones = 1.0 - (not_dones[np.arange(len(batch_inds)), increments]).reshape(len(batch_inds), 1)
        return sb3.common.type_aliases.DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(actions),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(dones),
            rewards=self.to_torch(self._normalize_reward(rewards)),
        )

TensorDict = Dict[Union[str, int], torch.Tensor]

class DictReplayBufferSamples(ReplayBufferSamples):
    observations: TensorDict
    actions: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    returns: torch.Tensor

class NStepLambdaDictReplayBuffer(sb3.common.buffers.DictReplayBuffer):
    """  
    Replay Buffer that computes N-step returns.
    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[torch.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    :param optimize_memory_usage: (bool) Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param n_steps: (int) The number of transitions to consider when computing n-step returns
    :param gamma:  (float) The discount factor for future rewards.
    """

    def __init__(
        self,
        buffer_size: int, 
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1, 
        optimize_memory_usage: bool = False,
        n_steps: int = 1, 
        gamma: float = 0.99,
        lmbda: float = 0.9
    ):   
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)
        self.n_steps = int(n_steps)
        if not 0 < n_steps <= buffer_size:
            raise ValueError("n_steps needs to be strictly smaller than buffer_size, and strictly larger than 0")
        self.gamma = gamma
        self.lmbda = lmbda

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> DictReplayBufferSamples:
        gamma = self.gamma
        lmbda = self.lmbda
        # Broadcasting turns 1dim arange matrix to 2 dimensional matrix that contains all
        # the indices, % buffersize keeps us in buffer range
        # indices is a [B x self.n_step ] matrix
        indices = (np.arange(self.n_steps) + batch_inds.reshape(-1, 1)) % self.buffer_size

        # two dim matrix of not dones. If done is true, then subsequent dones are turned to 0
        # using accumulate. This ensures that we don't use invalid transitions
        # not_dones is a [B x n_step] matrix
        not_dones = np.squeeze(1.0 - self.dones[indices], axis=-1)
        not_dones = np.multiply.accumulate(not_dones, axis=1)
        dones = 1.0 - not_dones
        # vector of the discount factor
        # [n_step] vector
        gammas = gamma ** np.arange(self.n_steps)
        lmbdas = lmbda ** np.arange(self.n_steps)
        # two dim matrix of rewards for the indices
        # using indices we select the current transition, plus the next n_step ones
        rewards = np.squeeze(self.rewards[indices], axis=-1)
        rewards = self._normalize_reward(rewards, env) * not_dones
        returns = np.expand_dims(
            np.sum(lmbdas * np.add.accumulate(gammas * rewards), -1) * (1 - lmbda), -1
        ).astype(np.float32)

        obs_ = self._normalize_obs({key: obs[batch_inds, 0, :] for key, obs in self.observations.items()})
        next_obs_ = self._normalize_obs({key: obs[indices, 0, :] for key, obs in self.next_observations.items()})

        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        actions = self.actions[batch_inds]

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(actions),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(dones),
            returns=self.to_torch(returns),
        )

class PrioritisedReplayBufferSamples(sb3.common.type_aliases.ReplayBufferSamples):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor
    weights: torch.Tensor

class PrioritisedReplayBuffer(sb3.common.buffers.ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.1,
        beta: float = 0.1
    ):
        super(PrioritisedReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs,
            optimize_memory_usage, handle_timeout_termination
        )
        self.empty = True
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._random_state = np.random.RandomState()

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
        priority = 1.0 if self.empty else self.priorities.max()

        if self.full:
            if priority > self.priorities.min():
                idx = self.priorities.argmin()
                self.priorities[idx] = priority
                self.observations[idx] = np.array(obs).copy()
                if self.optimize_memory_usage:
                    self.observations[(idx + 1) % self.buffer_size] = np.array(next_obs).copy()
                else:
                    self.next_observations[idx] = np.array(next_obs).copy()
                self.actions[idx] = np.array(action).copy()
                self.rewards[idx] = np.array(reward).copy()
                self.dones[idx] = np.array(done).copy()
                if self.handle_timeout_termination:
                    self.timeouts[idx] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
            else:
                pass # low priority experiences should not be included in buffer
        else:
            self.priorities[self.pos] = priority
            self.observations[self.pos] = np.array(obs).copy()
            if self.optimize_memory_usage:
                self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
            else:
                self.next_observations[self.pos] = np.array(next_obs).copy()
            self.actions[self.pos] = np.array(action).copy()
            self.rewards[self.pos] = np.array(reward).copy()
            self.dones[self.pos] = np.array(done).copy()
            self.pos += 1
            if self.handle_timeout_termination:
                self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        if self.pos >= 0 or self.full:
            self.empty = False
        if self.pos == self.buffer_size:
            self.full = True

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        """Update the priorities associated with particular experiences."""
        self.priorities[idxs] = priorities

    def sample(self, batch_size: int, env: Optional[sb3.common.vec_env.vec_normalize.VecNormalize] = None) -> PrioritisedReplayBufferSamples:
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
        ps = self.priorities[:self.pos]
        sampling_probs = ps**self.alpha / np.sum(ps**self.alpha)
        batch_inds = self._random_state.choice(
            np.arange(ps.size),
            size = batch_size,
            replace = True,
            p = sampling_probs
        )
        weights = (self.pos * sampling_probs[batch_inds]) ** -self.beta
        normalised_weights = weights / weights.max()
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        return PrioritisedReplayBufferSamples(
            observations = self.to_torch(self._normalize_obs(self.observations[batch_inds, 0, :], env)),
            actions = self.to_torch(self.actions[batch_inds]),
            next_observations = self.to_torch(next_obs),
            dones = self.to_torch(self.dones[batch_inds] * (1 - self.timeouts[batch_inds])),
            rewards = self.to_torch(self._normalize_reward(self.rewards[batch_inds], env)),
            weights = self.to_torch(normalised_weights)
        )

class PrioritisedTD3(sb3.td3.td3.TD3):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        """
            https://colab.research.google.com/github/davidrpugh/stochastic-expatriate-descent/blob/2020-04-14-prioritized-experience-replay/_notebooks/2020-04-14-prioritized-experience-replay.ipynb
        """

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
            deltas = sum(
                [target_q_values - current_q for current_q in current_q_values]
            ) / len(current_q_values)
            priorities = (deltas.abs()
                            .cpu()
                            .detach()
                            .numpy()
                            .flatten())
            self.replay_buffer.update_priorities(priorities)
            critic_loss = torch.mean((deltas * replay_data.weights)**2)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

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

class NStepHistoryVecTransposeImage(sb3.common.vec_env.base_vec_env.VecEnvWrapper):
    """
    Re-order channels, from HxWxC to CxHxW.
    It is required for PyTorch convolution layers.
    :param venv:
    """

    def __init__(self, venv: sb3.common.vec_env.base_vec_env.VecEnv, n_steps, image_space_keys = None):
        self.n_steps = n_steps
        observation_space = copy.deepcopy(venv.observation_space)
        if image_space_keys is None:
            self.image_space_keys = ['observation']
        else:
            self.image_space_keys = image_space_keys
        for key in self.image_space_keys:
            observation_space.spaces[key] = self.transpose_space(observation_space[key])
        super(NStepHistoryVecTransposeImage, self).__init__(venv, observation_space=observation_space)

    def transpose_space(self, observation_space: gym.spaces.Box, key: str = "") -> gym.spaces.Box:
        """
        Transpose an observation space (re-order channels).
        :param observation_space:
        :param key: In case of dictionary space, the key of the observation space.
        :return:
        """
        # Sanity checks
        assert not (observation_space.shape[1] == 3 or observation_space.shape[1] == 4), \
            "The observation space {key} must follow the channel last convention"
        height, width, channels = observation_space.shape
        new_shape = (channels, height, width)
        return gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=observation_space.dtype)

    def transpose_image(self, image: np.ndarray) -> np.ndarray:
        """
        Transpose an image or batch of images (re-order channels).
        :param image:
        :return:
        """
        if len(image.shape) == 3:
            return np.transpose(image, (2, 0, 1))
        return np.transpose(image, (0, 3, 1, 2))

    def transpose_observations(self, observations: Union[np.ndarray, Dict]) -> Union[np.ndarray, Dict]:
        """
        Transpose (if needed) and return new observations.
        :param observations:
        :return: Transposed observations
        """
        if isinstance(observations, dict):
            # Avoid modifying the original object in place
            observations = copy.deepcopy(observations)
            for k in self.image_space_keys:
                observations[k] = self.transpose_image(observations[k])
        else:
            observations = self.transpose_image(observations)
        return observations

    def step_wait(self) -> sb3.common.vec_env.base_vec_env.VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()

        # Transpose the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.transpose_observations(infos[idx]["terminal_observation"])

        return self.transpose_observations(observations), rewards, dones, infos

    def reset(self) -> Union[np.ndarray, Dict]:
        """
        Reset all environments
        """
        return self.transpose_observations(self.venv.reset())

    def close(self) -> None:
        self.venv.close()

class TD3HistoryPolicy(sb3.td3.policies.TD3Policy):
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
        n_steps: int,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        features_extractor_class: Type[sb3.common.torch_layers.BaseFeaturesExtractor] = sb3.common.torch_layers.FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        self.n_steps = n_steps
        super(TD3HistoryPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics = n_critics,
            share_features_extractor = share_features_extractor
        )

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if sb3.common.preprocessing.is_image_space(obs_space):
                    obs_ = np.stack([sb3.common.preprocessing.maybe_transpose(obs[:, i], obs_space) for i in range(self.n_steps)], 1)
                else:
                    obs_ = np.array(obs)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = np.stack([sb3.common.preprocessing.maybe_transpose(observation[:, i], obs_space) for i in range(self.n_steps)], 1)
            observation = observation.reshape((-1,) + self.observation_space.shape)
        else:
            observation = np.array(observation)
            observation = observation.reshape((-1,) + self.observation_space.shape)


        observation = sb3.common.utils.obs_as_tensor(observation, self.device)

        with torch.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
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


        return actions, state

class TD3SS(sb3.td3.td3.TD3):
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
        n_steps: float = 5,
        lmbda: float = 0.9,
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
        self.lmbda = lmbda
        self.n_steps = n_steps
        super(TD3SS, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,  # 1e6
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
            _init_setup_model
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
                actions = self.actor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(replay_data.observations, actions).mean()
                actor_loss += torch.nn.functional.mse_loss(actions, replay_data.observations['sampled_action'])
                actor_losses.append(actor_loss.item())

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


class TD3History(sb3.common.off_policy_algorithm.OffPolicyAlgorithm):
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
        n_steps: float = 5, 
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
        self.n_steps = n_steps
        self.history = None
        super(TD3History, self).__init__(
            policy,
            env,
            sb3.td3.policies.TD3Policy,
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
        super(TD3History, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

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
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

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

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[sb3.common.type_aliases.GymEnv],
        callback: sb3.common.type_aliases.MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, sb3.common.callbacks.BaseCallback]:
        """
        Initialize different variables needed for training.
        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()
        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()
        self.history = [copy.deepcopy(self._last_obs) for i in range(self.n_steps)]

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = sb3.common.utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)
        return total_timesteps, callback

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
            if isinstance(self._last_obs, dict):
                obs = {
                    key: np.stack(
                        [self.history[i][key] for i in range(len(self.history))], 1
                    ) for key in self._last_obs.keys()
                }
            else:
                obs = np.stack(self.history, 1)
            unscaled_action, _ = self.predict(obs, deterministic=False)

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
        )

        self._last_obs = new_obs
        self.history.pop(0)
        self.history.append(copy.deepcopy(self._last_obs))
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

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
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()
        return self

    def _excluded_save_params(self) -> List[str]:
        return super(TD3History, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

class TD3Lambda(sb3.td3.td3.TD3):
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
        n_steps: float = 5,
        lmbda: float = 0.9,
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
        self.lmbda = lmbda
        self.n_steps = n_steps
        super(TD3Lambda, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,  # 1e6
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
            _init_setup_model
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
                target_q_values = replay_data.returns
                for i in range(self.n_steps):
                    actions = replay_data.actions.clone()
                    noise = actions.data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    if isinstance(replay_data.observations, dict):
                        next_observations = {key: ob[:, i] for key, ob in replay_data.next_observations.items()}
                    else:
                        next_observations = replay_data.next_observations[:, i]
                    next_actions = (self.actor_target(next_observations) + noise).clamp(-1, 1)

                    next_q_values = torch.cat(self.critic_target(next_observations, next_actions), dim=1)
                    next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                    dones = torch.unsqueeze(replay_data.dones[:, i], -1)
                    out = dones * next_q_values
                    q_value = (1 - dones) * self.gamma * next_q_values * (self.lmbda ** i) * (1 - self.lmbda)
                    target_q_values += q_value
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
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

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
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[sb3.common.noise.ActionNoise] = None,
        replay_buffer_class: sb3.common.buffers.ReplayBuffer = DictReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = True,
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
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    """
    def _sample_action(
        self, learning_starts: int, action_noise: Optional[sb3.common.noise.ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = self._last_obs['sampled_action']
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None and self.num_timesteps > learning_starts:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        total_actor_losses, actor_losses, critic_losses, value_losses = [], [], [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions, next_vt, next_bg_out = self.actor_target(replay_data.next_observations)
                next_actions = (next_actions + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                target_v_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_vt

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
                actions, vt, bg_out = self.actor(replay_data.observations)
                value_loss = torch.nn.functional.mse_loss(vt, target_v_values)
                actor_loss = -self.critic.q1_forward(replay_data.observations, actions).mean()
                actor_losses.append(actor_loss.item())
                value_losses.append(value_loss.item())
                loss = actor_loss + value_loss
                total_actor_losses.append(loss.item())
                # Optimize the actor
                self.actor.optimizer.zero_grad()
                loss.backward()
                self.actor.optimizer.step()

                sb3.common.utils.polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                sb3.common.utils.polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if len(value_losses) > 0:
            self.logger.record("train/value_loss", np.mean(value_losses))
        if len(total_actor_losses) > 0:
            self.logger.record("train/total_actor_loss", np.mean(total_actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

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
        return super(TD3BG, self)._excluded_save_params() + ["actor", "actor_target", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

class ImitationLearning(sb3.common.off_policy_algorithm.OffPolicyAlgorithm):
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
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[sb3.common.noise.ActionNoise] = None,
        replay_buffer_class: sb3.common.buffers.DictReplayBuffer = sb3.common.buffers.DictReplayBuffer,
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
        super(ImitationLearning, self).__init__(
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
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        total_actor_losses, actor_losses, critic_losses, value_losses = [], [], [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                #noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                #noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions, next_vt = self.actor_target(replay_data.next_observations)
                #next_actions = (next_actions + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                target_v_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_vt

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
                actions, vt = self.actor(replay_data.observations)
                value_loss = torch.nn.functional.mse_loss(vt, target_v_values)
                actor_loss = -self.critic.q1_forward(replay_data.observations, actions).mean()
                actor_losses.append(actor_loss.item())
                value_losses.append(value_loss.item())
                loss = actor_loss + value_loss
                total_actor_losses.append(loss.item())
                # Optimize the actor
                self.actor.optimizer.zero_grad()
                loss.backward()
                self.actor.optimizer.step()

                sb3.common.utils.polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                sb3.common.utils.polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if len(value_losses) > 0:
            self.logger.record("train/value_loss", np.mean(value_losses))
        if len(total_actor_losses) > 0:
            self.logger.record("train/total_actor_loss", np.mean(total_actor_losses))
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
        return super(TD3BG, self)._excluded_save_params() + ["actor", "actor_target", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
