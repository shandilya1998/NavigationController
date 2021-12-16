from simulations.maze_env import MazeEnv
from simulations.collision_env import CollisionEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms, \
    GoalRewardNoObstacle
env = MazeEnv(PointEnv, GoalRewardNoObstacle)
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import shutil
import os

if os.path.exists(os.path.join('assets', 'plots', 'tests')):
    shutil.rmtree(os.path.join('assets', 'plots', 'tests'))
os.mkdir(os.path.join('assets', 'plots', 'tests'))

img = np.zeros(
    (200 * len(env._maze_structure), 200 * len(env._maze_structure[0])),
    dtype = np.float32
)

POS = []
OBS = []
REWARDS = []
INFO = []
IMAGES = []
done = False

steps = 0
pbar = tqdm()
count = 0
count_collisions = 0
count_ball = 0
ob = env.reset()

fig, ax = plt.subplots(1,1,figsize= (5,5))
line, = ax.plot(REWARDS, color = 'r', linestyle = '--')
ax.set_xlabel('steps')
ax.set_ylabel('reward')
total_reward = 0.0
while not done:
    ob, reward, done, info = env.step(ob['sampled_action'])
    if reward != 0.0:
        count += 1
    if info['collision_penalty'] != 0:
        count_collisions += 1
    if info['outer_reward'] > 0:
        count_ball += 1
    pbar.update(1)
    steps += 1
    pos = env.wrapped_env.sim.data.qpos.copy()    
    img = cv2.cvtColor(ob['observation'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join('assets', 'plots', 'tests', 'test_image_{}.png'.format(steps)), img)
    IMAGES.append(img)
    top = env.render('rgb_array')
    cv2.imshow('camera stream', img)
    cv2.imshow('position stream', top)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    POS.append(pos.copy())
    OBS.append(ob.copy())
    REWARDS.append(reward)
    total_reward += reward
    INFO.append(info)
    ax.clear()
    ax.plot(REWARDS, color = 'r', linestyle = '--')
    plt.pause(0.1)
pbar.close()
print('Ideal Path:')
print('total count:      {}'.format(count))
print('collision counts: {}'.format(count_collisions))
print('ball counts:      {}'.format(count_ball))
print('total_reward:     {}'.format(total_reward))
block_size = 50
fig2, ax = plt.subplots(1,1)
def xy_to_imgrowcol(x, y):
    (row, row_frac), (col, col_frac) = env._xy_to_rowcol_v2(x, y)
    row = block_size * row + int((row_frac) * block_size) + int(block_size / 2)
    col = block_size * col + int((col_frac) * block_size) + int(block_size / 2)
    return int(row), int(col)

img = np.zeros(
    (block_size * len(env._maze_structure), block_size * len(env._maze_structure[0]), 3)
)

for i in range(len(env._maze_structure)):
    for j in range(len(env._maze_structure[0])):
        if  env._maze_structure[i][j].is_wall_or_chasm():
            img[
                block_size * i: block_size * (i + 1),
                block_size * j: block_size * (j + 1)
            ] = 0.5

for i, goal in enumerate(env._task.goals):
    pos = goal.pos
    row, col = xy_to_imgrowcol(pos[0], pos[1]) 
    if i == env._task.goal_index:
        colors = [1, 0, 0]
    else:
        colors = [0, 1, 0]
    img[row - int(block_size / 10): row + int(block_size / 10), col - int(block_size / 10): col + int(block_size / 10)] = colors

"""
for index in range(len(env.sampled_path)):
    i, j = env._graph_to_structure_index(env.sampled_path[index])
    img[
        block_size * i + int(2 * block_size / 5): block_size * (i + 1) - int(2 * block_size / 5),
        block_size * j + int(2 * block_size / 5): block_size * (j + 1) - int(2 * block_size / 5)
    ] = [1, 0, 0]
    if index > 0:
        i_prev, j_prev = env._graph_to_structure_index(env.sampled_path[index - 1])
        delta_x = 1
        delta_y = 1
        if i_prev > i:
            delta_x = -1
        if j_prev > j:
            delta_y = -1
        x_points = np.arange(block_size * i_prev + int(block_size / 2), block_size * i + int(block_size / 2), delta_x, dtype = np.int32)
        y_points = np.arange(block_size * j_prev + int(block_size / 2), block_size * j + int(block_size / 2), delta_y, dtype = np.int32)
        if i_prev == i:
            x_points = np.array([block_size * i_prev + int(block_size / 2)] * block_size, dtype = np.int32)
        if j_prev == j:
            y_points = np.array([block_size * j_prev + int(block_size / 2)] * block_size, dtype = np.int32)
        for x, y in zip(x_points, y_points):
            img[x - int(block_size / 50): x + int(block_size / 50), y - int(block_size / 50): y + int(block_size / 50)] = [0, 1, 0]
"""

for pos in POS:
    row, col = xy_to_imgrowcol(pos[0], pos[1])
    img[row - int(block_size / 50): row + int(block_size / 50), col - int(block_size / 50): col + int(block_size / 50)] = [0, 0, 1]

ax.imshow(np.flipud(img))
fig.savefig('output.png')
plt.show()
