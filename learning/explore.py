import torch
import numpy as np
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import stable_baselines3 as sb3
from utils.td3_utils import TD3BG, TD3BGPolicy, DictReplayBuffer
from constants import params
from utils.callbacks import CustomCallback, CheckpointCallback, EvalCallback
import os
import shutil

torch.autograd.set_detect_anomaly(True)

class Explore:
    def __init__(self, logdir, max_episode_size):
        self.logdir = logdir
        self.env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
                lambda : sb3.common.monitor.Monitor(MazeEnv(
                    PointEnv,
                    CustomGoalReward4Rooms,
                    max_episode_size
                ))
            ]),
        )
        self.eval_env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
                lambda : sb3.common.monitor.Monitor(MazeEnv(
                    PointEnv,
                    CustomGoalReward4Rooms,
                    max_episode_size
                ))
            ]),
        )
        self.__set_rl_callback()
        n_actions = self.env.action_space.sample().shape[-1]
        
        self.rl_model = TD3BG(
            TD3BGPolicy,
            self.env,
            tensorboard_log = self.logdir,
            learning_rate = params['lr'],
            learning_starts = params['learning_starts'],
            batch_size = params['batch_size'],
            buffer_size = params['buffer_size'],
            gamma = params['gamma'],
            tau = params['tau'],
            gradient_steps = 2,
            train_freq = (10, 'step'),
            verbose = 2,
            device = 'auto'
        )

    def __set_rl_callback(self):
        recordcallback = CustomCallback(
            self.eval_env,
            render_freq = params['render_freq']
        )
        checkpointcallback = CheckpointCallback(
            save_freq = params['save_freq'],
            save_path = self.logdir,
            name_prefix = 'rl_model'
        )
        evalcallback = EvalCallback(
            self.eval_env,
            best_model_save_path = self.logdir,
            eval_freq = params['eval_freq'],
        )
        self.rl_callback = sb3.common.callbacks.CallbackList([
            checkpointcallback,
            evalcallback,
        ])

    def learn(self, timesteps):
        self.rl_model.learn(
            total_timesteps = timesteps,
            callback = self.rl_callback
        )
