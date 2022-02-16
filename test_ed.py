from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
from constants import params
import torch
from bg.autoencoder import Autoencoder
import stable_baselines3 as sb3

env = MazeEnv(PointEnv, CustomGoalReward4Rooms)

env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
    sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
        lambda : sb3.common.monitor.Monitor(env)  
    ])  
)

model = Autoencoder([1,1,1,1], 1000, 3)
inp = torch.zeros((1, 9, 64, 64))

print(inp.shape)
with torch.no_grad():
    out = model(inp)
if isinstance(out, tuple) or isinstance(out, list):
    for o in out:
        print(o.shape)
else:
    print(out.shape)
