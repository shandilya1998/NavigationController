from neurorobotics.simulations.maze_env import MazeEnv
from neurorobotics.simulations.point import PointEnv
from neurorobotics.simulations.maze_task import CustomGoalReward4Rooms
import cv2
import os
import shutil

def detect_objects(
        frame: np.ndarray):
    boxes = None
    infos = None
    return boxes, infos

def generate_data(
        datapath: str = 'neurorobotics/data/images',
        episodes: int = 100,
        max_episode_size: int = 2000):
    """Generates Object Detection Data in COCO format using `MazeEnv`.

    :param datapath: Path to folder data is written to
    :type datapath: str
    :param episodes: Number of episodes to run data generation for
    :type episodes: int
    :param max_episode_size: Maximum number of steps in an episode
    :type max_episode_size: int
        
    """
    if os.path.exists(datapath):
        shutil.rmtree(datapath)
    os.mkdir(datapath)

    env = MazeEnv(
        model_cls=PointEnv,
        maze_task=CustomGoalReward4Rooms,
        max_episode_size=max_episode_size
        mode='datagen')

    for ep in range(max_episode_size):
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, info = env.step(obs['sampled_action'])
        frame = obs['scale_1']
        
