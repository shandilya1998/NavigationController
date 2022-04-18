from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import matplotlib.pyplot as plt

env = MazeEnv(PointEnv, CustomGoalReward4Rooms)

obs = env.wrapped_env._get_obs()
depth, image = env._get_borders(obs['front_depth'], obs['front'])


fig, ax = plt.subplots(1, 2, figsize = (5, 10))
ax[0].imshow(depth, cmap = 'gray')
ax[1].imshow(image)

plt.show()
