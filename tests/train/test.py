from neurorobotics.train.td3 import train
from neurorobotics.simulations.point import PointEnv
from neurorobotics.simulations.maze_task import CustomGoalReward4Rooms
from neurorobotics.constants import params
from neurorobotics.simulations.maze_env import MazeEnv
import stable_baselines3 as sb3

train(
        logdir='assets/outputs/',
        env_class=MazeEnv,
        agent_class=PointEnv,
        task_class=CustomGoalReward4Rooms,
        policy_class=sb3.td3.MlpPolicy,
        params=params,
        )

