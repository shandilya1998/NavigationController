import torch
import stable_baselines3 as sb3
import gym
from neurorobotics.bg.autoencoder import Autoencoder

class TimeDistributedFeaturesExtractor(sb3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self,
        observation_space: gym.Space,
        features_dim: int,
        pretrained_params_path = 'assets/out/models/autoencoder/model.pt',
        device = None,
    ):
        super(TimeDistributedFeaturesExtractor, self).__init__(observation_space, features_dim)

        self.autoencoder = Autoencoder()
        self.autoencoder.load_state_dict(
            torch.load(pretrained_params_path, map_location=device)['model_state_dict']
        )
        self.autoencoder.eval()
        self.conv = torch.nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(512, features_dim),
            torch.nn.Tanh()
        )

        self.fc_sensors = torch.nn.Sequential(
            torch.nn.Linear(
                observation_space['sensors'].shape[-1],
                features_dim
            ),
            torch.nn.Tanh()
        )

        self.combine = torch.nn.Sequential(
            torch.nn.Linear(2 * features_dim, features_dim),
            torch.nn.Tanh()
        )

    def forward(self, observations):
        image = torch.cat([
            observations['scale_1'], observations['scale_2']
        ], 2)
        batch_size = image.size(0)
        seq_len = image.size(1)
        image_shape = image.shape[2:]
        image = image.view(-1, *image_shape)
        with torch.no_grad():
            visual, _ = self.autoencoder(image)
        visual = self.conv(visual)
        visual = visual.view(-1, visual.size(1))
        visual = self.linear(visual)

        sensors = observations['sensors'].view(-1, self._observation_space['sensors'].shape[-1])
        sensors = self.fc_sensors(sensors)

        features = torch.cat([visual, sensors], -1)
        features = self.combine(features)

        features = features.view(batch_size, seq_len, self.features_dim)

        return features



