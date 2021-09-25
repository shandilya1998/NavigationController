import numpy as np
import gym

def convert_observation_to_space(observation, maximum = float('inf')):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ])) 
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -maximum, dtype=np.float32)
        high = np.full(observation.shape, maximum, dtype=np.float32)
        space = gym.spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space
