from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import copy
import cv2
import numpy as np

env = MazeEnv(PointEnv, CustomGoalReward4Rooms)

import matplotlib.pyplot as plt

obs = env.reset()

FRAMES = []
REWARDS = []
INFRAMES = []

done = False
fig, ax = plt.subplots(2, 2, figsize = (10, 10))

def detect_target(frame):
    contours, hierarchy = env.detect_target(frame)
    bbx = []
    if len(contours):
        red_area = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(red_area)
        cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 0, 255), 1)
        bbx.extend([x, y, w, h])
    return bbx

def get_scales(frame, bbx):

    size = frame.shape[0]
    assert frame.shape[0] == frame.shape[1]
    scale_1 = frame.copy()
    scale_2 = frame.copy()
    if len(bbx) > 0:
        x, y, w, h = bbx

        # scale 1
        center_x, center_y = x + w // 2, y + h // 2
        x_min = center_x - size // (3 * 2)
        x_max = center_x + size // (3 * 2)
        y_min = center_y - size // (3 * 2)
        y_max = center_y + size // (3 * 2)
        if x_min < 0:
            center_x += np.abs(x_min)
            x_max += np.abs(x_min)
            x_min = 0 
        if x_max > size:
            offset = x_max - size
            center_x -= offset
            x_min -= offset
        if y_min < 0:
            center_y += np.abs(y_min)
            y_max += np.abs(y_min)
            y_min = 0
        if y_max > size:
            offset = y_max - size
            center_y -= offset
            y_min -= offset
        scale_1 = scale_1[y_min:y_max, x_min:x_max]

        # scale 2
        center_x, center_y = x + w // 2, y + h // 2
        x_min = center_x - 2 * size // (3 * 2)
        x_max = center_x + 2 * size // (3 * 2)
        y_min = center_y - 2 * size // (3 * 2)
        y_max = center_y + 2 * size // (3 * 2)
        if x_min < 0:
            center_x += np.abs(x_min)
            x_max += np.abs(x_min)
            x_min = 0
        if x_max > size:
            offset = x_max - size
            center_x -= offset
            x_min -= offset
        if y_min < 0:
            center_y += np.abs(y_min)
            y_max += np.abs(y_min)
            y_min = 0
        if y_max > size:
            offset = y_max - size
            center_y -= offset
            y_min -= offset
        scale_2 = scale_2[y_min:y_max, x_min:x_max]

    else:
        scale_1 = scale_1[
            size - size // (3 * 2): size + size // (3 * 2),
            size - size // (3 * 2): size + size // (3 * 2)
        ]

        scale_2 = scale_2[
            size - 2 * size // (3 * 2): size + 2 * size // (3 * 2),
            size - 2 * size // (3 * 2): size + 2 * size // (3 * 2)
        ]

    return scale_1, scale_2

while not done:
    obs, reward, done, info = env.step(obs['sampled_action'])
    frame = env.wrapped_env._get_obs()['front'].copy()
    real = frame.copy()
    ax[0][0].clear()
    ax[0][0].imshow(real)
    bbx = detect_target(frame)
    ax[0][1].clear()
    ax[0][1].imshow(frame)
    scale_1, scale_2 = get_scales(real, bbx)    
    ax[1][0].clear()
    ax[1][0].imshow(scale_1)
    ax[1][1].clear()
    ax[1][1].imshow(scale_2)
    REWARDS.append(copy.deepcopy(reward))
    INFRAMES.append(obs['inframe'])
    plt.pause(0.001)

