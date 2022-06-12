from neurorobotics.simulations.maze_env import MazeEnv, DiscreteMazeEnv
from neurorobotics.simulations.point import PointEnv, PointEnvV2
from neurorobotics.simulations.maze_task import CustomGoalReward4Rooms, \
    GoalRewardNoObstacle, GoalRewardSimple, CustomGoalReward4RoomsV2
env = DiscreteMazeEnv(PointEnv, CustomGoalReward4Rooms)
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import shutil
import os
from typing import Dict, List, NamedTuple, Optional, Tuple, Type
from neurorobotics.constants import params
import copy

class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"

RED = Rgb(0.7, 0.1, 0.1)
rgb = RED
def get_hsv_ranges(rgb):
    if rgb == RED:
        return (0, 25, 0), (15, 255, 255)
    elif rgb == GREEN:
        return (36,0,0), (86,255,255)
    elif rgb == BLUE:
        return (94, 80, 2), (126, 255, 255)
min_range, max_range = get_hsv_ranges(rgb)
#---------- Draw detected blobs: returns the image
#-- return(im_with_keypoints)
def draw_keypoints(image,                   #-- Input image
                   keypoints,               #-- CV keypoints
                   line_color=(0,0,255),    #-- line's color (b,g,r)
                   imshow=False             #-- show the result
                  ):  
    
    #-- Draw detected blobs as red circles.
    #-- cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), line_color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
    
    return(im_with_keypoints)

#---------- Apply search window: returns the image
#-- return(image)
def apply_search_window(image, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])    
    
    #--- Initialize the mask as a black image
    mask = np.zeros(image.shape,np.uint8)
    
    #--- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px,x_min_px:x_max_px] = image[y_min_px:y_max_px,x_min_px:x_max_px]   
    
    #--- return the mask
    return(mask)

#---------- Draw search window: returns the image
#-- return(image)
def draw_window(image,              #- Input image
                window_adim,        #- window in adimensional units
                color=(255,0,0),    #- line's color
                line=5,             #- line's thickness
                imshow=False        #- show the image
               ):

    rows = image.shape[0]
    cols = image.shape[1]

    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])

    #-- Draw a rectangle from top left to bottom right corner
    image = cv2.rectangle(image,(x_min_px,y_min_px),(x_max_px,y_max_px),color,line)

    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", image)

    return(image)

#---------- Draw X Y frame
#-- return(image)
def draw_frame(image,
               dimension=0.3,      #- dimension relative to frame size
               line=2              #- line's thickness
    ):  
    
    rows = image.shape[0]
    cols = image.shape[1]
    size = min([rows, cols])
    center_x = int(cols/2.0)
    center_y = int(rows/2.0)
    
    line_length = int(size*dimension)
    
    #-- X
    image = cv2.line(image, (center_x, center_y), (center_x+line_length, center_y), (0,0,255), line)
    #-- Y
    image = cv2.line(image, (center_x, center_y), (center_x, center_y+line_length), (0,255,0), line)
    
    return (image)

#---------- Apply a blur to the outside search region
#-- return(image)
def blur_outside(image, blur=5, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])    
    
    #--- Initialize the mask as a black image
    mask    = cv2.blur(image, (blur, blur))
    
    #--- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px,x_min_px:x_max_px] = image[y_min_px:y_max_px,x_min_px:x_max_px]   
    
    
    
    #--- return the mask
    return(mask)
    
#---------- Obtain the camera relative frame coordinate of one single keypoint
#-- return(x,y)
def get_blob_relative_position(image, keyPoint):
    rows = float(image.shape[0])
    cols = float(image.shape[1])
    # print(rows, cols)
    center_x    = 0.5*cols
    center_y    = 0.5*rows
    # print(center_x)
    x = (keyPoint.pt[0] - center_x)/(center_x)
    y = (keyPoint.pt[1] - center_y)/(center_y)
    return(x,y)

def blob_detect(image,                  #-- The frame (cv standard)
                hsv_min,                #-- minimum threshold of the hsv filter [h_min, s_min, v_min]
                hsv_max,                #-- maximum threshold of the hsv filter [h_max, s_max, v_max]
                blur=0,                 #-- blur value (default 0)
                blob_params=None,       #-- blob parameters (default None)
                search_window=None,     #-- window where to search as [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
                imshow=False
               ):


    #- Blur image to remove noise
    if blur > 0:
        image    = cv2.blur(image, (blur, blur))

    if imshow:
        cv2.imshow('blur', image)

    #- Search window
    if search_window is None: search_window = [0.0, 0.0, 1.0, 1.0]

    #- Convert image from BGR to HSV
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #- Apply HSV threshold
    mask    = cv2.inRange(hsv,hsv_min, hsv_max)

    #- Show HSV Mask
    if imshow:
        cv2.imshow("HSV Mask", mask)

    #- dilate makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=2)
    #- Show HSV Mask
    if imshow:
        cv2.imshow("Dilate Mask", mask)

    mask = cv2.erode(mask, None, iterations=2)

    #- Show dilate/erode mask
    if imshow:
        cv2.imshow("Erode Mask", mask)

    #- Cut the image using the search mask
    mask = apply_search_window(mask, search_window)

    if imshow:
        cv2.imshow("Searching Mask", mask)

    #- build default blob detection parameters, if none have been provided
    if blob_params is None:
        # Set up the SimpleBlobdetector with default parameters.
        params = cv2.SimpleBlobDetector_Params()
    
        # Change thresholds
        params.minThreshold = 0;
        params.maxThreshold = 100;
    
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 20000
    
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1 
    
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5 
    
        # Filter by Inertia
        params.filterByInertia =True
        params.minInertiaRatio = 0.5 
    
    else:
        params = blob_params    

    #- Apply blob detection
    detector = cv2.SimpleBlobDetector_create(params)

    # Reverse the mask: blobs are black on white
    reversemask = 255-mask
    
    if imshow:
        cv2.imshow("Reverse Mask", reversemask)
    
    keypoints = detector.detect(reversemask)

    return keypoints, reversemask

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
ob = env._get_obs()

"""
if params['debug']:
    fig, ax = plt.subplots(1,1,figsize= (3,3))
    line, = ax.plot(REWARDS, color = 'r', linestyle = '--')
    ax.set_xlabel('steps')
    ax.set_ylabel('reward')
"""

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
    """
    if params['debug']:
        cv2.imshow('observations', image)
        ax.clear()
        ax.plot(REWARDS, color = 'r', linestyle = '--')
        plt.pause(0.001)
    """

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
