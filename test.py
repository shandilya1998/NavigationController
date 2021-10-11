from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
env = MazeEnv(PointEnv, CustomGoalReward4Rooms)
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = np.zeros(
    (200 * len(env._maze_structure), 200 * len(env._maze_structure[0])),
    dtype = np.float32
)

X = []
Y = []

done = False

steps = 0
while not done or steps > 10000:
    try:
        ac = env.get_action()
    except IndexError:
        break
    ob, reward, done, info = env.step(ac)
    steps += 1

