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
from simulations.agent_model import AgentModel
from constants import params
from utils.env_utils import convert_observation_to_space
from collections import defaultdict, OrderedDict
import math
import cv2

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
        img = self._get_obs()
        dtype = img.dtype
        high = np.ones_like(img, dtype = dtype) * 255
        low = np.zeros_like(img, dtype = dtype)
        self.observation_space = gym.spaces.Box(
            low,
            high,
            shape = img.shape,
            dtype = dtype
        )

    def _set_action_space(self):
        low = np.array([0.0, -np.pi / 2], dtype = np.float32)
        high = np.array([self.VELOCITY_LIMITS * 1.41, np.pi / 2], dtype = np.float32)
        self.action_dim = 2
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        speed = action[0]
        yaw = action[1]
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)
        action = np.array([vx, vy, yaw], dtype = np.float32)
        self.sim.data.ctrl[:] = action
        for _ in range(0, self.frame_skip):
            self.sim.step()
        next_obs = self._get_obs()
        reward = np.sum(np.square(self.data.qvel[:2] / self.VELOCITY_LIMITS)) / 2
        reward = -np.square(self.data.qvel[2]) * 5e-3
        return next_obs, reward, False, {}

    def gaussian(self, x, mean, std):
        return np.exp(-0.5 * ((x - mean) / std) ** 2)

    def _get_obs(self):
        rgb, depth = self.sim.render(
            width = 100,
            height = 75,
            camera_name = 'mtdcam',
            depth = True
        )
        depth = 255 * (depth - 0.97) / 0.03
        depth = depth.astype(np.uint8)
        #cv2.imshow('depth', np.flipud(depth / 255.0))
        img = np.flipud(np.concatenate([
            rgb, np.expand_dims(depth, -1)
        ], -1))
        return img

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
        return self.sim.data.qpos[self.ORI_IND]

    def get_v(self):
        return np.linalg.norm(self.sim.data.qvel[:2])

    def set_ori(self, ori):
        qpos = self.sim.data.qpos.copy()
        qpos[self.ORI_IND] = ori
        self.set_state(qpos, self.data.qvel)
