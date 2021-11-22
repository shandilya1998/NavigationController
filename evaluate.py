import stable_baselines3 as sb3
import torch
import gym
import argparse
from utils.callbacks import evaluate_policy
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import os
from utils.td3_utils import TD3BG
import cv2
import skvideo.io as skv

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

    env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
        sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
            lambda : sb3.common.monitor.Monitor(MazeEnv(
                PointEnv,
                CustomGoalReward4Rooms,
                max_episode_size
            ))  
        ]), 
    )
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
        screen = env.render(mode="rgb_array")
        # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
        screens.append(cv2.cvtColor(screen.transpose(2, 0, 1), cv2.RBG2BGR))

    model_path = os.path.join(args.logdir, args.model_file)
    model = TD3BG.load(
        path = model_path,
        env = env,
        device = 'auto',
        print_system_info=True
    )

    evaluate_policy(
        model,
        env,
        callback = grab_screens,
        n_eval_episodes = 4,
        deterministic = True,
    )

    out_video = np.stack(screens, 0)

    skvideo.io.vwrite('{}_evaluation.mp4'.format(model_path), out_video)
