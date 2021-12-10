import torch
import numpy as np
from simulations.maze_env import MazeEnv
from simulations.collision_env import CollisionEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import stable_baselines3 as sb3
from utils.td3_utils import TD3BG, TD3BGPolicy, \
    DictReplayBuffer, TD3BGPolicyV2, HistoryFeaturesExtractor
from constants import params
from utils.callbacks import CustomCallback, CheckpointCallback, EvalCallback
import os
import shutil

torch.autograd.set_detect_anomaly(True)

class Explore:
    def __init__(self, logdir, max_episode_size, policy_version, env_type = 'maze', n_steps = 4):
        if env_type == 'maze':
            env_class = MazeEnv
        elif env_type == 'collision':
            env_class = CollisionEnv
        self.logdir = logdir
        self.env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
                lambda : sb3.common.monitor.Monitor(env_class(
                    PointEnv,
                    CustomGoalReward4Rooms,
                    max_episode_size,
                    policy_version
                ))
            ]),
        )
        self.eval_env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
                lambda : sb3.common.monitor.Monitor(env_class(
                    PointEnv,
                    CustomGoalReward4Rooms,
                    max_episode_size,
                    policy_version
                ))
            ]),
        )
        self.__set_rl_callback()
        n_actions = self.env.action_space.sample().shape[-1]

        model = TD3BG
        policy_kwargs = None
        if policy_version == 1:
            policy_class = TD3BGPolicy
            action_noise = None
        elif policy_version == 2:
            policy_class = TD3BGPolicyV2
            action_noise = sb3.common.noise.OrnsteinUhlenbeckActionNoise(
                params['OU_MEAN'] * np.ones(n_actions),
                params['OU_SIGMA'] * np.ones(n_actions)
            )
        elif policy_version == 3:
            model = sb3.TD3
            policy_class = 'CnnPolicy'
            action_noise = sb3.common.noise.OrnsteinUhlenbeckActionNoise(
                params['OU_MEAN'] * np.ones(n_actions),
                params['OU_SIGMA'] * np.ones(n_actions)
            )
        elif policy_version == 4:
            model = sb3.TD3
            policy_class = 'MlpPolicy'
            action_noise = sb3.common.noise.OrnsteinUhlenbeckActionNoise(
                params['OU_MEAN'] * np.ones(n_actions),
                params['OU_SIGMA'] * np.ones(n_actions)
            )
            policy_kwargs = {
                'feature_extractor_class' : HistoryFeatureExtractor
            }
        else:
            raise ValueError('Expected policy version less than or equal to 2, got {}'.format(policy_version))

        self.rl_model = model(
            policy_class,
            self.env,
            tensorboard_log = self.logdir,
            learning_rate = params['lr'],
            learning_starts = params['learning_starts'],
            batch_size = params['batch_size'],
            buffer_size = params['buffer_size'],
            action_noise = action_noise,
            optimize_memory_usage = True,
            gamma = params['gamma'],
            tau = params['tau'],
            train_freq = (1, 'episode'),
            verbose = 2,
            device = 'auto',
            policy_kwargs = policy_kwargs
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
