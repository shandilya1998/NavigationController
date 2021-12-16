import torch
import numpy as np
from simulations.maze_env import MazeEnv
from simulations.collision_env import CollisionEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms, \
    GoalRewardNoObstacle, GoalRewardSimple
import stable_baselines3 as sb3
from utils.td3_utils import TD3BG, TD3BGPolicy, \
    DictReplayBuffer, TD3BGPolicyV2, HistoryFeaturesExtractor, \
    MultiModalFeaturesExtractor, NStepReplayBuffer, \
    NStepDictReplayBuffer, TD3Lambda, \
    NStepLambdaDictReplayBuffer, NStepLambdaReplayBuffer
from constants import params
from utils.callbacks import CustomCallback, CheckpointCallback, EvalCallback
import os
import shutil
import gym

torch.autograd.set_detect_anomaly(True)

def linear_schedule(initial_value, final_value):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * (initial_value - final_value) + final_value

    return func

class Explore:
    def __init__(self,
        logdir,
        max_episode_size,
        policy_version,
        env_type = 'maze',
        history_steps = 4,
        task_version = 1,
        n_steps = 10,
        lmbda = 0.9
    ):
        if env_type == 'maze':
            env_class = MazeEnv
            print('Env Type: maze')
        elif env_type == 'collision':
            env_class = CollisionEnv
            print('Env Type: collision')
            raise ValueError
        if task_version == 1:
            task = CustomGoalReward4Rooms
            print('Task: CustomGoalReward4Rooms')
        elif task_version == 2:
            task = GoalRewardNoObstacle
            print('Task: GoalRewardNoObstacle')
        elif task_version == 3:
            task = GoalRewardSimple
            print('Task: GoalRewardSimple')
        self.logdir = logdir
        self.env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
                lambda : sb3.common.monitor.Monitor(env_class(
                    PointEnv,
                    task,
                    max_episode_size,
                    policy_version,
                    history_steps
                ))
            ])
        )

        self.eval_env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
                lambda : sb3.common.monitor.Monitor(env_class(
                    PointEnv,
                    task,
                    max_episode_size,
                    policy_version,
                    history_steps
                ))
            ])
        )
        
        self.__set_rl_callback()
        n_actions = self.env.action_space.sample().shape[-1]

        model = TD3BG
        kwargs = {}
        policy_kwargs = None
        optimize_memory_usage = True
        replay_buffer_class = None
        replay_buffer_kwargs = None
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
                'features_extractor_class' : HistoryFeaturesExtractor
            }
        elif policy_version == 5:
            model = sb3.TD3
            policy_class = 'MlpPolicy'
            action_noise = sb3.common.noise.OrnsteinUhlenbeckActionNoise(
                params['OU_MEAN'] * np.ones(n_actions),
                params['OU_SIGMA'] * np.ones(n_actions)
            )
            policy_kwargs = {
                'features_extractor_class' : MultiModalFeaturesExtractor
            }
            optimize_memory_usage = False
        else:
            raise ValueError('Expected policy version less than or equal to 2, got {}'.format(policy_version))

        if n_steps > 0:
            replay_buffer_class = NStepReplayBuffer
            replay_buffer_kwargs = {
                'n_steps' : n_steps
            }
            if isinstance(self.env.observation_space, gym.spaces.dict.Dict):
                replay_buffer_class = NStepDictReplayBuffer

        if lmbda > 0 and lmbda < 1 and n_steps > 0:
            print('Using TD(λ) learning')
            replay_buffer_class = NStepLambdaReplayBuffer
            if isinstance(self.env.observation_space, gym.spaces.dict.Dict):
                replay_buffer_class = NStepLambdaDictReplayBuffer
            replay_buffer_kwargs = { 
                'n_steps' : n_steps,
                'lmbda' : lmbda
            }
            model = TD3Lambda
            kwargs['lmbda'] = lmbda
            kwargs['n_steps'] = n_steps

        print('Model: {}'.format(model))
        self.rl_model = model(
            policy_class,
            self.env,
            tensorboard_log = self.logdir,
            learning_rate = linear_schedule(params['lr'], params['final_lr']),
            learning_starts = params['learning_starts'],
            batch_size = params['batch_size'],
            buffer_size = params['buffer_size'],
            action_noise = action_noise,
            optimize_memory_usage = optimize_memory_usage,
            gamma = params['gamma'],
            tau = params['tau'],
            train_freq = (1, 'episode'),
            replay_buffer_class = replay_buffer_class,
            replay_buffer_kwargs = replay_buffer_kwargs,
            verbose = 2,
            device = 'auto',
            policy_kwargs = policy_kwargs,
            **kwargs
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
