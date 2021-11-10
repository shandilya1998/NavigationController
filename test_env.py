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

POS = []
OBS = []
REWARDS = []
INFO = []
done = False

steps = 0
while not done:
    ac = env.get_action()
    ob, reward, done, info = env.step(ac)
    steps += 1
    pos = env.wrapped_env.sim.data.qpos.copy()    
    img = cv2.cvtColor(ob['observation'], cv2.COLOR_RGB2BGR)
    cv2.imshow('stream', img)
    cv2.waitKey(1)
    POS.append(pos.copy())
    OBS.append(ob.copy())
    REWARDS.append(reward)
    INFO.append(info)
    #env.render()

img = np.zeros(
    (200 * len(env._maze_structure), 200 * len(env._maze_structure[0]), 3)
)

for i in range(len(env._maze_structure)):
    for j in range(len(env._maze_structure[0])):
        if  env._maze_structure[i][j].is_wall_or_chasm():
            img[
                200 * i: 200 * (i + 1),
                200 * j: 200 * (j + 1)
            ] = 0.5


def xy_to_imgrowcol(x, y):
    (row, row_frac), (col, col_frac) = env._xy_to_rowcol_v2(x, y)
    row = 200 * row + int((row_frac) * 200) + 100
    col = 200 * col + int((col_frac) * 200) + 100
    return row, col

for index in range(len(env.sampled_path)):
    i, j = env._graph_to_structure_index(env.sampled_path[index])
    img[
        200 * i + 80: 200 * (i + 1) - 80,
        200 * j + 80: 200 * (j + 1) - 80
    ] = [1, 0, 0]
    if index > 0:
        i_prev, j_prev = env._graph_to_structure_index(env.sampled_path[index - 1])
        delta_x = 1
        delta_y = 1
        if i_prev > i:
            delta_x = -1
        if j_prev > j:
            delta_y = -1
        x_points = np.arange(200 * i_prev + 100, 200 * i + 100, delta_x, dtype = np.int32)
        y_points = np.arange(200 * j_prev + 100, 200 * j + 100, delta_y, dtype = np.int32)
        if i_prev == i:
            x_points = np.array([200 * i_prev + 100] * 200, dtype = np.int32)
        if j_prev == j:
            y_points = np.array([200 * j_prev + 100] * 200, dtype = np.int32)
        for x, y in zip(x_points, y_points):
            img[x - 4: x + 4, y - 4: y + 4] = [0, 1, 0]


for pos in POS:
    row, col = xy_to_imgrowcol(pos[0], pos[1])
    img[row - 4: row + 4, col - 4: col + 4] = [0, 0, 1]

plt.imshow(np.flipud(img))
plt.show()
