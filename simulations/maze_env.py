import numpy as np
import gym
import os

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()
