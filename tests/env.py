from neurorobotics.simulations.maze_env import MazeEnv
from neurorobotics.simulations.point import PointEnv
from neurorobotics.simulations.maze_task import create_simple_room_maze
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import shutil
import os
from typing import Dict, List, NamedTuple, Optional, Tuple, Type
from neurorobotics.constants import params
import copy
from neurorobotics.utils.cv_utils import *
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to run tests on environments.")
    parser.add_argument(
            '--env',
            help='name of the supported environment to use',
            type=str
    )
    args = parser.parse_args()

    task_generator = None
    if args.env == 'SimpleRoom':
        task_generator = create_simple_room_maze

    assert task_generator is not None

    env = MazeEnv(PointEnv, task_generator)

    if os.path.exists(os.path.join('neurorobotics', 'assets', 'plots', 'tests')):
        shutil.rmtree(os.path.join('neurorobotics', 'assets', 'plots', 'tests'))
    os.mkdir(os.path.join('neurorobotics', 'assets', 'plots', 'tests'))

    img = np.zeros(
        (200 * len(env._maze_structure), 200 * len(env._maze_structure[0])),
        dtype=np.float32
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
    ob = env._get_obs()

    total_reward = 0.0
    ac = env.get_action()
    top = env.render('rgb_array')
    image_size = (3 * top.shape[0] , 2 * top.shape[1])
    video = cv2.VideoWriter(
        'test_env.avi',
        cv2.VideoWriter_fourcc(*"MJPG"), 10, image_size, isColor = True
    )

    speed = []
    angular = []
    while not done:
        ob, reward, done, info = env.step(ob['sampled_action'])
        speed.append(np.sqrt(env.data.qvel[0] ** 2 + env.data.qvel[1] ** 2))
        angular.append(env.data.qvel[2])
        top = env.render('rgb_array')
        top = cv2.cvtColor(
            top,
            cv2.COLOR_RGB2BGR
        )
        scale_1 = cv2.resize(ob['scale_1'], top.shape[:2])
        scale_2 = cv2.resize(ob['scale_2'], top.shape[:2])
        depth = np.repeat(ob['depth'].transpose(1, 2, 0), 3, 2) * 255
        depth = cv2.resize(depth, top.shape[:2])
        loc_map = cv2.resize(ob['loc_map'], top.shape[:2])
        prev_loc_map = cv2.resize(ob['prev_loc_map'], top.shape[:2])

        image = np.concatenate([
            np.concatenate([
                scale_1, scale_2
            ], 0),
            np.concatenate([
                depth, top
            ], 0),
            np.concatenate([
                prev_loc_map, loc_map
            ], 0)
        ], 1).astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)

        #ob, reward, done, info = env.step(env.action_space.sample())
        ac = env.get_action()
        if reward != 0.0:
            count += 1
        if info['collision_penalty'] != 0:
            count_collisions += 1
        pbar.update(1)
        steps += 1
        pos = env.wrapped_env.sim.data.qpos.copy()    
        POS.append(pos.copy())
        OBS.append(ob.copy())
        REWARDS.append(reward)
        total_reward += reward
        INFO.append(copy.deepcopy(info))

    video.release()
    pbar.close()
#plt.close()
    print('Ideal Path:')
    print('collision counts: {}'.format(count_collisions))
    print('total_reward:     {}'.format(total_reward))

    block_size = 50
    _REWARDS = copy.deepcopy(REWARDS)
    REWARDS = np.array(REWARDS, dtype = np.float32)
    REWARDS = np.clip(REWARDS, a_min = -0.020, a_max = 0.020)
    fig2, ax = plt.subplots(4,4, figsize = (24, 24))
    ax[0][1].set_xlabel('steps')
    ax[0][1].set_ylabel('speed')
    ax[1][0].set_xlabel('steps')
    ax[1][0].set_ylabel('angular velocity')
    ax[1][1].set_xlabel('steps')
    ax[1][1].set_ylabel('reward')
    ax[0][2].set_xlabel('steps')
    ax[0][2].set_ylabel('coverage reward')
    coverage_reward = np.array([
        info['coverage_reward'] for info in INFO
    ], dtype = np.float32)
    coverage_reward = np.clip(coverage_reward, a_min = -0.020, a_max = 0.020)
    ax[0][2].plot(coverage_reward)
    ax[0][3].set_xlabel('steps')
    ax[0][3].set_ylabel('inner reward')
    inner_reward = np.array([
        info['inner_reward'] for info in INFO
    ], dtype = np.float32)
    inner_reward = np.clip(inner_reward, a_min = -0.020, a_max = 0.020)
    ax[0][3].plot(inner_reward)
    ax[1][2].set_xlabel('steps')
    ax[1][2].set_ylabel('collision penalty')
    collision_penalty = np.array([
        info['collision_penalty'] for info in INFO
    ], dtype = np.float32)
    collision_penalty = np.clip(collision_penalty, a_min = -0.020, a_max = 0.020)
    ax[1][2].plot(collision_penalty)
    ax[1][3].set_xlabel('steps')
    ax[1][3].set_ylabel('outer reward')
    outer_reward = np.array([
        info['outer_reward'] for info in INFO
    ], dtype = np.float32)
    outer_reward = np.clip(outer_reward, a_min = -0.020, a_max = 0.020)
    ax[1][3].plot(outer_reward)
    ax[2][0].set_xlabel('steps')
    ax[2][0].set_ylabel('reward - coverage reward')
    items = REWARDS - coverage_reward
    ax[2][0].plot(items)
    ax[2][1].set_xlabel('steps')
    ax[2][1].set_ylabel('reward - inner reward')
    items = REWARDS - inner_reward
    ax[2][1].plot(items)
    ax[2][2].set_xlabel('steps')
    ax[2][2].set_ylabel('reward - collision penalty')
    items = REWARDS - collision_penalty
    ax[2][2].plot(items)
    ax[2][3].set_xlabel('steps')
    ax[2][3].set_ylabel('reward - outer reward')
    items = REWARDS - outer_reward
    ax[2][3].plot(items)
    ax[3][0].set_xlabel('steps')
    ax[3][0].set_ylabel('coverage, inner reward')
    ax[3][0].plot(coverage_reward, label = 'coverage reward')
    ax[3][0].plot(inner_reward, label = 'inner reward')
    ax[3][0].legend()
    ax[3][0].set_xlabel('steps')
    ax[3][1].set_ylabel('coverage, outer reward')
    ax[3][1].plot(coverage_reward, label = 'coverage reward')
    ax[3][1].plot(outer_reward, label = 'outer reward')
    ax[3][1].legend()
    ax[3][2].set_xlabel('steps')
    ax[3][2].set_ylabel('inner, outer reward')
    ax[3][2].plot(inner_reward, label = 'inner reward')
    ax[3][2].plot(outer_reward, label = 'outer reward')
    ax[3][2].legend()
    ax[3][3].set_xlabel('steps')
    ax[3][3].set_ylabel('q value')
    q = []
    _REWARDS = np.array(_REWARDS, dtype = np.float32)
    for i in range(len(REWARDS)):
        gamma = np.arange(len(_REWARDS) - i)
        gamma = params['gamma'] ** gamma
        q.append(np.sum(_REWARDS[i:] * gamma))
    q = np.array(q, dtype = np.float32)
    ax[3][3].plot(q)


    def xy_to_imgrowcol(x, y):
        (row, row_frac), (col, col_frac) = env._xy_to_rowcol_v2(x, y)
        row = block_size * row + int((row_frac) * block_size)
        col = block_size * col + int((col_frac) * block_size)
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

    for wx, wy in zip(env.wx, env.wy):
        row, col = xy_to_imgrowcol(wx, wy) 
        img[row - int(block_size / 10): row + int(block_size / 10), col - int(block_size / 10): col + int(block_size / 10)] = [1, 1, 0]

    for i, goal in enumerate(env._task.objects):
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

    for x, y in zip(env.cx, env.cy):
        row, col = xy_to_imgrowcol(x, y)
        img[row - int(block_size / 50): row + int(block_size / 50), col - int(block_size / 50): col + int(block_size / 50)] = [1, 1, 1]



    ax[0][0].imshow(np.rot90(np.flipud(img)))
    ax[0][1].plot(speed)
    ax[1][0].plot(angular)
    ax[1][1].plot(REWARDS)
    plt.tight_layout()
    fig2.savefig('output.png')
    plt.show()
