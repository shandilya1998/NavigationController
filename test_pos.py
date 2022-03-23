from simulations.maze_env import MazeEnv
from simulations.maze_task import CustomGoalReward4Rooms
from simulations.point import PointEnv
import matplotlib.pyplot as plt
import cv2
import numpy as np 

env = MazeEnv(PointEnv, CustomGoalReward4Rooms, mode = 'eval')

"""
for i, struct in enumerate(env._maze_structure):
    for j, s in enumerate(struct):
        print(i,j,s)
"""

top = env.render('rgb_array')

while True:
    cv2.imshow('top', top)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


