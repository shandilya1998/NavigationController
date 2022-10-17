import stable_baselines3 as sb3
import numpy as np
import torch
import argparse
from neurorobotics.train.td3 import train
from neurorobotics.simulations.maze_env import SimpleRoomEnv, LocalPlannerEnv
from neurorobotics.simulations.maze_task import create_simple_room_maze, create_local_planner_area
from neurorobotics.simulations.point import PointEnv, BlindPointEnv
from neurorobotics.constants import params
from neurorobotics.utils.schedules import linear_schedule
from neurorobotics.utils.feature_extractors import DictToTensorFeaturesExtractor, LocalPlannerFeaturesExtractor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Script for TD3 for SimpleRoomEnv')
    parser.add_argument(
        '--logdir',
        type=str,
        help='Path to logging directory'
    )
    parser.add_argument(
        '--env',
        type=str,
        help='Environment to use for training td3 policy'

    )
    args = parser.parse_args()

    env_class = None
    agent_class = None
    task_generator = None
    features_extractor_class = None
    if args.env == 'SimpleRoom':
        env_class = SimpleRoomEnv
        agent_class = PointEnv
        task_generator = create_simple_room_maze
        features_extractor_class = DictToTensorFeaturesExtractor
    elif args.env == 'LocalPlanner':
        env_class = LocalPlannerEnv
        agent_class = BlindPointEnv
        task_generator = create_local_planner_area
        features_extractor_class = LocalPlannerFeaturesExtractor

    action_noise = sb3.common.noise.OrnsteinUhlenbeckActionNoise(
        params['OU_MEAN'] * np.ones(SimpleRoomEnv.n_actions),
        params['OU_SIGMA'] * np.ones(SimpleRoomEnv.n_actions),
        dt=params['dt']
    )

    policy_kwargs = {
        'net_arch': params['net_arch'],
        'activation_fn': torch.nn.ReLU,
        'features_extractor_class': features_extractor_class,
        'features_extractor_kwargs': {
            'features_dim': params['num_ctx'],
        },
        'normalize_images': True,
        'optimizer_class': torch.optim.Adam,
        'optimizer_kwargs': None,
        'n_critics': params['n_critics'],
        'share_features_extractor': True
    }

    model = train(
        env_class=env_class,
        agent_class=agent_class,
        task_generator=task_generator,
        policy_class='MultiInputPolicy',
        params=params,
        lr_schedule=linear_schedule(params['lr'], params['final_lr']),
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        device='auto',
        logdir=args.logdir
    )
