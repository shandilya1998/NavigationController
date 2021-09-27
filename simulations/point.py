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
    MANUAL_COLLISION: bool = True
    RADIUS: float = 0.4

    VELOCITY_LIMITS: float = 10.0

    def __init__(self, file_path: Optional[str] = 'point.xml') -> None:
        file_path = os.path.join(
            os.getcwd(),
            'assets',
            'xml',
            file_path
        )
        super().__init__(file_path, 1)
        high = np.inf * np.ones(6, dtype=np.float32)
        high[3:] = self.VELOCITY_LIMITS * 1.2
        high[self.ORI_IND] = np.pi
        low = -high
        self.observation_space = gym.spaces.Box(low, high)


    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        """
        self.action_dim = params['input_size_low_level_control']
        low = -1.2 * np.ones(self.action_dim, dtype = np.float32) * self.VELOCITY_LIMITS
        high = -low
        """
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        qpos = self.sim.data.qpos.copy()
        qpos[2] += action[-1] * self.dt * self.frame_skip
        penalty = 0.0
        if np.any(np.abs(action[2:-1]) > 0.0):
            penalty += -1.0
        # Clip orientation
        if qpos[2] < -np.pi:
            qpos[2] += np.pi * 2
        elif np.pi < qpos[2]:
            qpos[2] -= np.pi * 2
        # Compute increment in each direction
        qpos[0] += action[0] * self.dt
        qpos[1] += action[1] * self.dt
        qvel = self.sim.data.qvel.copy()
        #qvel[0] = action[0]
        #qvel[1] = action[1]
        #qvel[2] = action[-1]
        qvel = np.clip(qvel, -self.VELOCITY_LIMITS, self.VELOCITY_LIMITS)
        self.set_state(qpos, qvel)
        #ac = np.array([action[0], action[1], action[-1]], dtype = np.float32)
        """
        self.sim.data.ctrl[:] = action
        for _ in range(0, self.frame_skip):
            self.sim.step()
        next_obs = self._get_obs()
        return next_obs, 0.0, False, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:3],  # Only point-relevant coords.
                self.sim.data.qvel.flat[:3],
            ]
        )

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

    def set_xy(self, xy: np.ndarray) -> None:
        qpos = self.sim.data.qpos.copy()
        qpos[:2] = xy
        self.set_state(qpos, self.sim.data.qvel)

    def get_ori(self):
        return self.sim.data.qpos[self.ORI_IND]
