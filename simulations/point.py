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

class PointEnv(AgentModel):
    FILE: str = "point.xml"
    ORI_IND: int = 2
    RADIUS: float = 0.4
    VELOCITY_LIMITS: float = 12.0

    def __init__(self, file_path: Optional[str] = 'point.xml') -> None:
        file_path = os.path.join(
            os.getcwd(),
            'assets',
            'xml',
            file_path
        )
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
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_dim = low.shape[-1]
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.sim.data.ctrl[:] = action
        prev_pos = self.get_xy().copy()
        for _ in range(0, self.frame_skip):
            self.sim.step()
        next_obs = self._get_obs()
        pos = self.get_xy().copy()
        reward = np.sum(np.square(self.data.qvel[:2] / self.VELOCITY_LIMITS * 1.2)) / 2
        return next_obs, 0.0, False, {}

    def _get_obs(self):
        rgb, depth = self.sim.render(
            width = 100,
            height = 75,
            camera_name = 'mtdcam',
            depth = True
        )
        depth = 255 * ((depth - depth.min()) / (depth.max() - depth.min()))
        depth = depth.astype(np.uint8)
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
