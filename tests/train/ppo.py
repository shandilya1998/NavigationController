import stable_baselines3 as sb3
import numpy as np
import torch
import argparse
from neurorobotics.train.ppo import train
from neurorobotics.simulations.maze_env import SimpleRoomEnv
from neurorobotics.simulations.maze_task import create_simple_room_maze
from neurorobotics.simulations.point import PointEnv
from neurorobotics.constants import params
from neurorobotics.utils.schedules import linear_schedule
from neurorobotics.utils.feature_extractors import DictToTensorFeaturesExtractor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Script for PPO for SimpleRoomEnv')
    parser.add_argument(
        '--logdir',
        type=str,
        help='Path to logging directory'
    )
    args = parser.parse_args()

    action_noise = sb3.common.noise.OrnsteinUhlenbeckActionNoise(
        params['OU_MEAN'] * np.ones(SimpleRoomEnv.n_actions),
        params['OU_SIGMA'] * np.ones(SimpleRoomEnv.n_actions),
        dt=params['dt']
    )

    policy_kwargs = {
        'net_arch': params['net_arch'],
        'activation_fn': torch.nn.ReLU,
        'ortho_init': True,
        'features_extractor_class': DictToTensorFeaturesExtractor,
        'features_extractor_kwargs': {
            'features_dim': params['num_ctx'],
        },
        'normalize_images': True,
        'optimizer_class': torch.optim.Adam,
        'optimizer_kwargs': None,
    }

    model = train(
        env_class=SimpleRoomEnv,
        agent_class=PointEnv,
        task_generator=create_simple_room_maze,
        policy_class='MultiInputPolicy',
        params=params,
        lr_schedule=linear_schedule(params['lr'], params['final_lr']),
        policy_kwargs=policy_kwargs,
        device='auto',
        logdir=args.logdir
    )
