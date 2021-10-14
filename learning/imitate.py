import torch
import numpy as np
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import stable_baselines3 as sb3
from utils.il_utils import ImitationLearning
from constants import params
from utils.callbacks import CustomCallback
import os
import shutil

class Imitate:
    def __init__(self, logdir, datapath):
        self.logdir = logdir
        self.datapath = datapath
        self.env = MazeEnv(PointEnv, CustomGoalReward4Rooms)
        self.eval_env = MazeEnv(PointEnv, CustomGoalReward4Rooms)
        self.__set_il_callback()
        self.il_model = ImitationLearning(
            'MlpPolicy',
            sb3.common.monitor.Monitor(self.env),
            tensorboard_log = os.path.join(self.logdir, 'tensorboard'),
            learning_starts = params['learning_starts'],
            train_freq = (5, "step"),
            verbose = 2,
            batch_size = params['batch_size'],
            buffer_size = params['buffer_size'],
            policy_kwargs = {
                'features_extractor_class' : sb3.common.torch_layers.NatureCNN
            }
        )

    def __set_il_callback(self):
        recordcallback = CustomCallback(
            self.eval_env,
            render_freq = params['render_freq']
        )
        if os.path.exists(os.path.join(
            self.logdir, 'checkpoints'
        )):
            shutil.rmtree(os.path.join(
                self.logdir, 'checkpoints'
            ))
        os.mkdir(os.path.join(
                self.logdir, 'checkpoints'
        ))
        checkpointcallback = sb3.common.callbacks.CheckpointCallback(
            save_freq = params['save_freq'],
            save_path = os.path.join(self.logdir, 'checkpoints'),
            name_prefix = 'il_model'
        )
        if os.path.exists(os.path.join(
            self.logdir, 'best_model'
        )):
            shutil.rmtree(os.path.join(
                self.logdir,
                'best_model'
            ))
        os.mkdir(os.path.join(
            self.logdir,
            'best_model'
        ))
        evalcallback = sb3.common.callbacks.EvalCallback(
            self.eval_env,
            best_model_save_path = os.path.join(
                self.logdir, 'best_model'
            ),
            eval_freq = params['eval_freq'],
            log_path = self.logdir
        )
        self.il_callback = sb3.common.callbacks.CallbackList([
            checkpointcallback,
            recordcallback,
            evalcallback,
        ])

    def learn(self):
        self.il_model.learn(
            total_timesteps = params['total_timesteps'],
            callback = self.il_callback
        )
