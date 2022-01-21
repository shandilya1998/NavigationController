from typing import Any, Dict, List, Optional, Tuple, Type, Union, NamedTuple
import stable_baselines3 as sb3
import torch
import gym
import argparse
from utils.callbacks import evaluate_policy
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms, GoalRewardNoObstacle
import os
from utils.td3_utils import TD3BG
import cv2
import skvideo.io as skv
import numpy as np
from constants import params
from utils.rtd3_utils import RTD3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir',
        type = str,
        help = 'relative path to directory with data'
    )
    parser.add_argument(
        '--model_file',
        type = str,
        help = 'name of the model file to load in log director'
    )
    parser.add_argument(
        '--max_episode_size',
        type = int,
        help = 'maximum episode length'
    )
    parser.add_argument(
        '--history_steps',
        type = int,
        help = 'number of images in observation input to policy',
        default = 4
    )
    args = parser.parse_args()

    env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
        sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
            lambda : sb3.common.monitor.Monitor(MazeEnv(
                PointEnv,
                CustomGoalReward4Rooms,
                args.max_episode_size,
                args.history_steps
            ))  
        ]), 
    )
    model_path = os.path.join(args.logdir, args.model_file)
    top = env.render(mode="rgb_array").shape
    obs = env.observation_space['front'].shape
    video1 = cv2.VideoWriter(
        '{}_evaluation_camera_view.avi'.format(model_path),
        cv2.VideoWriter_fourcc(*"MJPG"), 10, (obs[2], obs[1]), isColor = True
    )
    video2 = cv2.VideoWriter(
        '{}_evaluation_top_view.avi'.format(model_path),
        cv2.VideoWriter_fourcc(*"MJPG"), 10, (top[0], top[1]), isColor = True
    )
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
        screen = env.render(mode="rgb_array")
        # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
        front = _locals['observations']['front'][0, :3].transpose(1, 2, 0)
        #back = _locals['observations']['back'][0, :3].transpose(1, 2, 0)
        #left = _locals['observations']['left'][0, :3].transpose(1, 2, 0)
        #right = _locals['observations']['right'][0, :3].transpose(1, 2, 0)
        #print(front.shape)
        #observation = np.concatenate([
        #    np.concatenate([front, back], 0),
        #    np.concatenate([left, right], 0)
        #], 1)
        observation = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
        video1.write(observation)
        video2.write(screen)

    model = RTD3.load(
        path = model_path,
        env = env,
        device = 'auto',
        print_system_info=True,
        custom_objects = {
            'hidden_state': [
            (torch.zeros((1, size)).to('cpu'), torch.zeros((1, size)).to('cpu')) \
                for size in [400, 300]
            ]
        }
    )
    evaluate_policy(
        model,
        env,
        callback = grab_screens,
        n_eval_episodes = 5,
        deterministic = True,
    )
    cv2.destroyAllWindows()
    video1.release()
    video2.release()
