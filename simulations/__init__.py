"""
Mujoco Maze
-----------

A maze environment using mujoco that supports custom tasks and robots.
"""


import gym
from neurorobotics.simulations.maze_task import TaskRegistry
from neurorobotics.simulations.point import PointEnv

for maze_id in TaskRegistry.keys():
    for i, task_gen in enumerate(TaskRegistry.tasks(maze_id)):
        """
            Need to find smarter way to implement task registration here using task_generators
            instead of using task classes.
        """
        gym.envs.register(
            id=f"{maze_id}-v{i}",
            entry_point="mujoco_maze.maze_env:MazeEnv",
            kwargs=dict(
                model_cls=PointEnv,
                maze_task_generator=task_gen,
                maze_size_scaling=4.0,
                inner_reward_scaling=1.0,
            ),
            max_episode_steps=1000,
            reward_threshold=1.0,
        )

__version__ = "2.0"
