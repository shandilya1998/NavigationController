from neurorobotics.train.td3 import train
from neurorobotics.simulations.point import PointEnv
from neurorobotics.simulations.maze_task import create_simple_room_maze
from neurorobotics.constants import params
from neurorobotics.simulations.maze_env import SimpleRoomEnv
import stable_baselines3 as sb3


def linear_schedule(initial_value, final_value):
    """ Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining):
        """ Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * (initial_value - final_value) + final_value

    return func

action_noise_kwargs = {
        'mean': params['OU_MEAN'],
        'sigma': params['OU_SIGMA'],
        'theta': params['OU_THETA'],
        'dt': params['dt'],
        }

train(
        logdir='assets/outputs/',
        env_class=SimpleRoomEnv,
        agent_class=PointEnv,
        task_generator=create_simple_room_maze,
        policy_class=sb3.td3.MlpPolicy,
        params=params,
        lr_schedule=linear_schedule(params['lr'], params['final_lr']),
        action_noise_class=sb3.common.noise.OrnsteinUhlenbeckActionNoise,
        action_noise_kwargs=action_noise_kwargs,
        policy_kwargs={}
        )
