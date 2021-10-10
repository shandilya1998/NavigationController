from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
env = MazeEnv(PointEnv, CustomGoalReward4Rooms)
while True:
    ac = env.get_action()
    ob, reward, done, info = env.step(ac)
    env.render()
