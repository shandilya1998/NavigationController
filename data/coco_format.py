from neurorobotics.simulations.maze_env import MazeEnv
from neurorobotics.simulations.point import PointEnv
from neurorobotics.simulations.maze_task import CustomGoalReward4Rooms
from neurorobotics.constants import params
import cv2
import os
import shutil
import argparse


def generate_data(
        datapath: str = 'neurorobotics/data/images',
        episodes: int = 1,
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
    if params['debug']:
        os.mkdir(os.path.join(datapath, 'top'))

    env = MazeEnv(
        model_cls=PointEnv,
        maze_task=CustomGoalReward4Rooms,
        max_episode_size=max_episode_size,
        mode='datagen')

    for ep in range(episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            obs, reward, done, info = env.step(obs['sampled_action'])
            frame = cv2.cvtColor(obs['scale_1'], cv2.COLOR_BGR2RGB)
            #boxes, info = env.detect_color(frame, False)
            top = env.render('rgb_array')
            top = cv2.cvtColor(
                top,
                cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(os.path.join(
                    datapath,
                    'image_ep_{}_step_{}.png'.format(ep + 1, step)), frame)
            if params['debug']:
                cv2.imwrite(os.path.join(
                        datapath,
                        'top/image_ep_{}_step_{}.png'.format(ep, step)), top)
            step += 1
        print("Episode {} Steps Done {}".format(ep, step))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to generate data in Coco format from MazeEnv."
    )
    parser.add_argument(
        '--datapath',
        type=str,
        help='Path to folder to store generated data in. Ensure folder contains no important data.'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        help='Number of episodes to generate data for.'
    )
    parser.add_argument(
        '--max_episode_size',
        type=int,
        help='Maximum number of steps to run environment simulations for. Note that this is the maximum number of steps allowed in an episode'
    )
    args = parser.parse_args()

    generate_data(
        datapath=args.datapath,
        episodes=args.num_episodes,
        max_episode_size=args.max_episode_size
    )
