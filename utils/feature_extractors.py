import stable_baselines3 as sb3
import gym
import torch
import numpy as np
from typing import Dict


class DictToTensorFeaturesExtractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    """Feature Extractor for `MlpPolicy` and similar policy architectures with a simple tensor
    input. Combines Visual Input with Proprioreceptive Input.

    :param observation_space: Observation Configuration.
    :type observation_space: gym.Space
    :param features_dim: Output 1D Tensor Dimension
    :type features_dim: int
    """
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int
    ):
        super(DictToTensorFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, kernel_size=8, stride=4, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 64, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=4, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        with torch.no_grad():
            inp = observation_space.sample()
            frame_t = torch.as_tensor(inp['frame_t'].astype(np.float32))[None]
            out = self.cnn(frame_t)
            sensors_dim = inp['sensors'].shape[-1]
            n_flatten = out.shape[-1]  # + positions.shape[-1]

        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(n_flatten + sensors_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, features_dim),
                torch.nn.ReLU()
        )

    def forward(
            self,
            observations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        visual_f = self.cnn(observations['frame_t'])
        inp = torch.cat([observations['sensors'], visual_f], -1)
        features = self.mlp(inp)
        return features


class LocalPlannerFeaturesExtractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    """Feature Extractor for `MlpPolicy` and similar policy architectures with a simple tensor
    input. Combines Proprioreceptive Input with Goal Information.

    :param observation_space: Observation Configuration.
    :type observation_space: gym.Space
    :param features_dim: Output 1D Tensor Dimension
    :type features_dim: int
    """
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            features_dim: int
    ):
        super(LocalPlannerFeaturesExtractor, self).__init__(observation_space, features_dim)
        input_dim = observation_space['sensors'].shape[-1] + \
            observation_space['achieved_goal'].shape[-1] + \
            observation_space['desired_goal'].shape[-1]
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, features_dim),
            torch.nn.ReLU()
        )

    def forward(
            self,
            observations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = torch.cat([
                observations['sensors'],
                observations['achieved_goal'],
                observations['desired_goal']
                ], -1)
        return self.mlp(x)
