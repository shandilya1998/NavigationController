import cv2
from neurorobotics.simulations.maze_env import MazeEnv
from neurorobotics.simulations.point import PointEnv
from neurorobotics.simulations.maze_task import CustomGoalReward4Rooms
from constants import params
import colorsys

def  detect_color(
        frame: np.ndarray
        ):
    boxes = []
    info = []
    for rgb in params['available_rgb']:
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        hsv_low = []
        hsv_high = []
        if h > 10 / 180:
            hsv_low.append(h * 180 - 10)
        else:
            hsv_low.append(0)
        if h < 160 / 180:
            hsv_high.append(h * 180 + 10)
        else:
            hsv_high.append(180)
        if s > 100 / 255:
            hsv_low.append(s * 255 - 100)
        else:
            hsv_low.append(0)
        if s < 155 / 255:
            hsv_high.append(s * 255 + 100)
        else:
            hsv_high.append(255)
        hsv_low.append(0)
        hsv_high.append(255)
        mask = cv2.inRange(frame, hsv_low, hsv_high)
        cv2.imshow('mask', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return boxes, info

if __name__ == '__main__':
    env = MazeEnv(
            model_cls=PointEnv,
            maze_task=CustomGoalReward4Rooms,
            max_episode_size=params['max_episode_size'])

    ob = env.reset()
    done = False
    while not done:
        ob, reward, done, info = env.step(ob['sampled_action'])
        boxes, info = detect_color(ob['scale_1'])
        
