import cv2
import numpy as np
from neurorobotics.simulations.maze_env import MazeEnv
from neurorobotics.simulations.point import PointEnv
from neurorobotics.simulations.maze_task import CustomGoalReward4Rooms
from neurorobotics.constants import params
import colorsys


def detect_color(
        frame: np.ndarray,
        display: bool = False
        ):
    boxes = []
    info = []
    boxes = []
    results = None
    masks = {}
    rgbs = params['available_rgb'] + [params['target_rgb']]
    for i, rgb in enumerate(rgbs):
        h, s, _ = colorsys.rgb_to_hsv(*rgb)
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
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(
                hsv_frame,
                np.array(hsv_low, dtype=np.int32),
                np.array(hsv_high, dtype=np.int32))
        if display:
            masks['r: {:.2f}, g: {:.2f}, b: {:.2f}'.format(
                    rgb[0],
                    rgb[1],
                    rgb[2]
                    )] = mask.copy()
            if results is None:
                results = mask
            else:
                results += mask
        contours, _ = cv2.findContours(
                mask.copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        bbx = []
        if len(contours):
            color_area = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(color_area)
            if display:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(
                        frame,
                        "class: {}".format(i),
                        (x + w, y + h + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 1, cv2.LINE_AA)
            bbx.extend([x, y, w, h])
            boxes.append(bbx)
            info.append(i)

    if display:
        results = np.clip(results, 0, 255).astype(np.uint8)
        cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imshow('mask', results)
        """
        for key, mask in masks.items():
            cv2.imshow(key, mask)
        """
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
        boxes, info = detect_color(ob['scale_1'], True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

