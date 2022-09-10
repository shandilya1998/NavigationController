from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np
import stable_baselines3 as sb3
import gym
import torch
from neurorobotics.utils.rtd3 import TimeDistributedFeaturesExtractor

class Network(torch.nn.Module):
    def __init__(
        self,
        features_dim,
        activation_fn,
        output_dim,
        net_arch,
        squash_output=False
    ):
        super(ActorNetwork, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=features_dim,
            hidden_size=net_arch[0],
            num_layers=1,
            bias=False,
            batch_first=True,
        )
        input_size = net_arch[0]
        linear = []
        for param in net_arch[1:]:
            linear.append(torch.nn.Linear(
                input_size,
                param
            ))
            linear.append(activation_fn())
            input_size = param
        
        self.backbone = torch.nn.Sequential(
            *linear
        )

        q = []
        for dim in output_dim:
            net = []
            net.append(torch.nn.Linear(input_size, dim))
            last = -1
            if squash_output:
                net.append(torch.nn.Tanh())
                last = -2
            torch.nn.init.uniform_(net[last].weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(net[last].bias, -3e-4, 3e-4)
            q.append(torch.nn.Sequential(*net))
        self.q = torch.nn.ModuleList(q)

    def forward(self, features, hidden_state):
        output, state = self.lstm(features, hidden_state)
        hidden_state = state
        output = self.linear(output[:, -1])
        q = []

        return [output, hidden_state]

class QNetwork(sb3.common.policies.BasePolicy):
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
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        normalize_images: bool = True,
    ):
        super(QNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        action_dim = self.action_space.n  # number of actions
        q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.q_net = torch.nn.Sequential(*q_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict the q-values.
        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs))

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

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
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

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

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QNetwork(**net_args).to(self.device)

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
