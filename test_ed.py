from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
from constants import params
import torch
from bg.models import Autoencoder

env = MazeEnv(PointEnv, CustomGoalReward4Rooms)

model = Autoencoder(env.observation_space['front'], 512)

inp = torch.zeros((1, 3, 75, 100))

with torch.no_grad():
    out = model(inp)
if isinstance(out, tuple) or isinstance(out, list):
    for o in out:
        print(o.shape)
else:
    print(out.shape)
