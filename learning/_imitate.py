import torch
import numpy as np
from neurorobotics.simulations.maze_env import MazeEnv
from neurorobotics.simulations.point import PointEnv
from neurorobotics.simulations.maze_task import CustomGoalReward4Rooms
import stable_baselines3 as sb3
from neurorobotics.utils.il_utils import ImitationLearning
from neurorobotics.utils.td3_utils import TD3BGPolicy
from neurorobotics.constants import params
from neurorobotics.utils.callbacks import CustomCallback, CheckpointCallback, EvalCallback
import os
import shutil

class Imitate:
    def __init__(self, logdir, batch_size, max_episode_size):
        self.logdir = logdir
        self.batch_size = batch_size
        self.env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
                lambda : sb3.common.monitor.Monitor(MazeEnv(
                    PointEnv,
                    CustomGoalReward4Rooms,
                    max_episode_size
                ))
            ])
        )
        self.eval_env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
                lambda : sb3.common.monitor.Monitor(MazeEnv(
                    PointEnv,
                    CustomGoalReward4Rooms,
                    max_episode_size
                ))
            ])
        )
        self.__set_il_callback()
        self.il_model = ImitationLearning(
            TD3BGPolicy,
            self.env,
            tensorboard_log = self.logdir,
            learning_starts = params['learning_starts'],
            train_freq = (5, "step"),
            verbose = 2,
            batch_size = self.batch_size,
            buffer_size = params['buffer_size'],
            policy_kwargs = {
                'features_extractor_class' : sb3.common.torch_layers.NatureCNN
            },
            device = 'auto'
        )

    def __set_il_callback(self):
        recordcallback = CustomCallback(
            self.eval_env,
            render_freq = params['render_freq']
        )
        checkpointcallback = CheckpointCallback(
            save_freq = params['save_freq'],
            save_path = self.logdir,
            name_prefix = 'il_model'
        )
        evalcallback = EvalCallback(
            self.eval_env,
            best_model_save_path = self.logdir,
            eval_freq = params['eval_freq'],
        )
        self.il_callback = sb3.common.callbacks.CallbackList([
            checkpointcallback,
            recordcallback,
            evalcallback,
        ])

    def learn(self, timesteps):
        self.il_model.learn(
            total_timesteps = timesteps,
            callback = self.il_callback
        )
