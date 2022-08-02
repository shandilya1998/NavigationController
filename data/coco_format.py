from neurorobotics.simulations.maze_env import MazeEnv
from neurorobotics.simulations.point import PointEnv
from neurorobotics.simulations.maze_task import CustomGoalReward4Rooms
import cv2
import os
import shutil
from tqdm import tqdm


def generate_data(
        datapath: str = 'neurorobotics/data/images',
        episodes: int = 2,
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
        max_episode_size=max_episode_size,
        mode='datagen')

    for ep in tqdm(range(episodes), position = 1):
        obs = env.reset()
        done = False
        step = 0
        bar = tqdm(total = max_episode_size, position = 0)
        while not done:
            obs, reward, done, info = env.step(obs['sampled_action'])
            frame = cv2.cvtColor(obs['scale_1'], cv2.COLOR_BGR2RGB)
            #boxes, info = env.detect_color(frame, False)
            cv2.imwrite(os.path.join(
                    datapath,
                    'image_ep_{}_step_{}.png'.format(ep, step)), frame)
            step += 1
            bar.update(1)
        bar.close()

if __name__ == '__main__':
    generate_data()
