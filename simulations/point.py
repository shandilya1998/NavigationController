"""
A ball-like robot as an explorer in the maze.
Based on `models`_ and `rllab`_.

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _rllab: https://github.com/rll/rllab
"""

from typing import Optional, Tuple
import gym
import numpy as np
import os
from neurorobotics.simulations.agent_model import AgentModel
from neurorobotics.constants import params
from neurorobotics.utils.env_utils import convert_observation_to_space
from collections import defaultdict, OrderedDict
import math
import cv2
from neurorobotics.constants import image_width, image_height

class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0
    
    def push(self, x):
        self.n += 1
    
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
        
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0
    
    def standard_deviation(self):
        return math.sqrt(self.variance())

class PointEnv(AgentModel):
    FILE: str = "point.xml"
    ORI_IND: int = 2
    RADIUS: float = 0.4
    VELOCITY_LIMITS: float = 10.0

    def __init__(self, file_path: Optional[str] = 'point.xml') -> None:
        file_path = os.path.join(
            os.getcwd(),
            'assets',
            'xml',
            file_path
        )
        self.rs1 = RunningStats()
        self.rs2 = RunningStats()
        super().__init__(file_path, 1)
        obs = self._get_obs()
        spaces = {}
        for key, item in obs.items():
            dtype = item.dtype
            high = np.ones_like(item, dtype = dtype) * 255
            low = np.zeros_like(item, dtype = dtype)
            spaces[key] = gym.spaces.Box(
                low, high, shape = item.shape, dtype = dtype
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def _set_action_space(self):
        # Modify def _setup_model(**kwargs) in class RTD3 if action space is modified
        low = np.array([0.0, -params['max_vyaw']], dtype = np.float32)
        high = np.array([self.VELOCITY_LIMITS * 1.41, params['max_vyaw']], dtype = np.float32)
        self.action_dim = 2
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    """
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    """

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # _vx and _vy are parallel and perpendicular to direction of motion respectively
        v = action[0]
        yaw = self.get_ori() 
        vyaw = action[1]
        yaw += vyaw * self.dt
        # vx and vy are along the x and y axes respectively
        vx = v * np.cos(yaw)
        vy = v * np.sin(yaw)
        action = np.array([vx, vy, vyaw], dtype = np.float32)
        self.sim.data.ctrl[:] = action
        """
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel
        qpos[self.ORI_IND] = yaw
        qpos[0] += vx * self.dt
        qpos[1] += vy * self.dt
        self.set_state(qpos, qvel)
        """
        for _ in range(0, self.frame_skip):
            self.sim.step()
        next_obs = self._get_obs()
        reward = np.linalg.norm(self.data.qvel[:2]) * 7.5e-2
        reward += -5e-3 * np.abs(self.data.qvel[self.ORI_IND])
        return next_obs, 0.0, False, {}

    def gaussian(self, x, mean, std):
        return np.exp(-0.5 * ((x - mean) / std) ** 2)

    def get_z(self, zbuffer):
        znorm = 2 * zbuffer - 1
        return -2 * self.model.vis.map.znear * self.model.vis.map.zfar / ((self.model.vis.map.zfar - self.model.vis.map.znear) * znorm - self.model.vis.map.znear - self.model.vis.map.zfar)

    def _get_obs(self):
        rgb1, depth1 = self.sim.render(
            width = image_width,
            height = image_height,
            camera_name = 'mtdcam1',
            depth = True 
        )
        """
        #depth1 = 255 * (depth - 0.68) / 0.32
        #depth1 = 255 * depth
        #depth1 = np.flipud(depth1.astype(np.uint8))
        #cv2.imshow('depth', depth1)
        #print(depth.max(), depth.min())
        rgb2 = self.sim.render(
            width = 100,
            height = 75,
            camera_name = 'mtdcam2',
            depth = False
        )
        rgb3 = self.sim.render(
            width = 100,
            height = 75, 
            camera_name = 'mtdcam3',
            depth = False
        )
        rgb4 = self.sim.render(
            width = 100,
            height = 75, 
            camera_name = 'mtdcam4',
            depth = False
        )
        depth1 = 255 * (depth1 - 0.965) / 0.035
        depth1 = depth1.astype(np.uint8)
        depth2 = 255 * (depth1 - 0.965) / 0.035
        depth2 = depth1.astype(np.uint8)
        depth3 = 255 * (depth1 - 0.965) / 0.035
        depth3 = depth1.astype(np.uint8)
        depth4 = 255 * (depth1 - 0.965) / 0.035
        depth4 = depth1.astype(np.uint8)
        img1 = np.flipud(np.concatenate([
            rgb1, np.expand_dims(depth1, -1)
        ], -1))
        img2 = np.flipud(np.concatenate([
            rgb2, np.expand_dims(depth2, -1)
        ], -1))
        img3 = np.flipud(np.concatenate([
            rgb3, np.expand_dims(depth3, -1)
        ], -1))
        img4 = np.flipud(np.concatenate([
            rgb4, np.expand_dims(depth4, -1)
        ], -1))
        """
        obs = {
            'front' : np.flipud(rgb1),
            'front_depth' : np.flipud(depth1)    
        }
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.sim.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * 0.1

        # Set everything other than point to original position and 0 velocity.
        qpos[3:] = self.init_qpos[3:]
        qvel[3:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_xy(self):
        return self.sim.data.qpos[:2].copy()

    def get_speed(self):
        return np.linalg.norm(self.sim.data.qvel[:2].copy())

    def set_xy(self, xy: np.ndarray) -> None:
        qpos = self.sim.data.qpos.copy()
        qpos[:2] = xy
        self.set_state(qpos, self.sim.data.qvel)

    def get_ori(self):
        ori = self.sim.data.qpos[self.ORI_IND]
        ori -= int(ori / (2 * np.pi)) * 2 * np.pi
        if ori > np.pi:
            ori -= 2 * np.pi
        elif ori < -np.pi:
            ori += 2 * np.pi
        return ori

    def get_v(self):
        return np.linalg.norm(self.sim.data.qvel[:2])

    def set_ori(self, ori):
        qpos = self.sim.data.qpos.copy()
        qpos[self.ORI_IND] = ori
        self.set_state(qpos, self.data.qvel)


class PointEnvV2(PointEnv):
    def __init__(self, file_path: Optional[str] = 'point.xml') -> None:
        super(PointEnvV2, self).__init__(file_path)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        speed = action[0]
        yaw = action[1]
        dx = speed * np.cos(yaw) * self.dt
        dy = speed * np.sin(yaw) * self.dt
        qpos = self.sim.data.qpos.copy()
        next_qpos = np.array([
            qpos[0] + dx,
            qpos[1] + dy,
            yaw
        ], dtype = np.float32)
        qvel = self.sim.data.qvel.copy()
        self.set_state(next_qpos, qvel)
        for _ in range(0, self.frame_skip):
            self.sim.step()
        next_obs = self._get_obs()
        reward = np.sum(np.square(self.data.qvel[:2] / self.VELOCITY_LIMITS)) / 2
        reward = -np.square(self.data.qvel[2]) * 1e-3
        return next_obs, reward, False, {}
