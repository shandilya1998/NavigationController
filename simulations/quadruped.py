import numpy as np
import gym
from typing import Optional, Tuple, Union
from neurorobotics.simulations.agent_model import AgentModel
from neurorobotics.constants import params, image_height, image_width
from neurorobotics.utils.env_utils import convert_observation_to_space


class MePedEnv(AgentModel):
    """Base Class for MePed Quadruped Simulation. This class ports simulations.QuadrupedV3 from
    Omnidirectional Controller.

    :param file_path: Path to agent xml description file.
    :type file_path: str
    :param frame_skip: Number of simulation steps per environment step.
    :type frame_skip: int
    """
    FILE: str = "meped.xml"
    ORI_IND: int = 2
    RADIUS: float = 0.4
    VELOCITY_LIMITS: float = 10.0

    def __init__(self, file_path: str, frame_skip: int) -> None:
        super().__init__(file_path, frame_skip)
        obs = self._get_obs()
        spaces = {}
        for key, item in obs.items():
            dtype = item.dtype
            high = np.ones_like(item, dtype=dtype) * 255
            low = np.zeros_like(item, dtype=dtype)
            spaces[key] = gym.spaces.Box(
                low, high, shape=item.shape, dtype=dtype
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def _set_action_space(self):
        low = np.array([0.0, -params['max_vyaw']], dtype=np.float32)
        high = np.array([self.VELOCITY_LIMITS * 1.41, params['max_vyaw']], dtype=np.float32)
        self.action_dim = 2
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def step(self, action: np.ndarray) -> Tuple[Union[np.ndarray, dict], float, bool, dict]:
        return super().step(action)

    def _get_obs(self):
        rgb1, depth1 = self.sim.render(
            width=image_width,
            height=image_height,
            camera_name='mtdcam1',
            depth=True 
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
