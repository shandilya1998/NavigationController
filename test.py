from simulations.maze_env import MazeEnv
from simulations.maze_task import CustomGoalReward4Rooms
from simulations.point import PointEnv
from tqdm import tqdm

env = MazeEnv(PointEnv, CustomGoalReward4Rooms, mode = 'imitate')

bar = tqdm()

while True:
    done = False
    ob = env.reset()
    while not done:
        ob, reward, done, info = env.step(ob['sampled_action'])
    bar.update(1)

bar.close()
