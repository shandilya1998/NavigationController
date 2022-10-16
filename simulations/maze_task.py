"""Maze tasks that are defined by their map, termination condition, and goals.
"""
import cv2
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union
import numpy as np
from neurorobotics.utils.cv_utils import blob_detect
from neurorobotics.simulations.maze_env_utils import MazeCell
import copy
from neurorobotics.constants import params
import colorsys
import random
import itertools as it


E, B, C, R, M = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.CHASM, MazeCell.ROBOT, MazeCell.XY_BLOCK


class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"


RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)


def get_hsv_ranges(rgb):
    if rgb == RED:
        return (0, 25, 0), (15, 255, 255)
    elif rgb == GREEN:
        return (36, 0, 0), (86, 255, 255)
    elif rgb == BLUE:
        return (94, 80, 2), (126, 255, 255)


def find_robot(structure, size_scaling):
    for i, j in it.product(range(len(structure)), range(len(structure[0]))):
        if structure[i][j].is_robot():
            return j * size_scaling, i * size_scaling
    raise ValueError("No robot in maze specification.")


def check_target_object_distance(agent, target):
    """Method to check if target is placed in the vicinity of the agent.

    :param target: Target row and column.
    :type target: Tuple[int, int]
    :param agent: Agent row and column.
    :type agent: Tuple[int, int]
    """
    arow, acol = agent
    trow, tcol = target
    if arow == trow:  # or arow + 1 == trow or arow - 1 == trow:
        return True
    if acol == tcol:  # or acol + 1 == tcol or acol - 1 == tcol:
        return True
    if arow + 1 == trow and acol + 1 == tcol:
        return True
    if arow - 1 == trow and acol + 1 == tcol:
        return True
    if arow + 1 == trow and acol - 1 == tcol:
        return True
    if arow - 1 == trow and acol - 1 == tcol:
        return True

    return False


class MazeObject:
    def __init__(
        self,
        pos: np.ndarray,
        characteristics: Dict[str, Union[int, float, np.ndarray]],
        reward_scale: float = 1.0,
    ) -> None:
        assert 0.0 <= reward_scale <= 1.0
        self.pos = pos
        self.dim = pos.shape[0]
        self.reward_scale = reward_scale
        self.rgb = characteristics['rgb']
        self.threshold = characteristics['threshold']
        self.custom_size = characteristics['size']
        self.site_type = characteristics['site_type']
        self.min_range = np.array(characteristics['hsv_low'], dtype=np.float32)
        self.max_range = np.array(characteristics['hsv_high'], dtype=np.float32)
        self.is_target = characteristics['target']
        self.kind = 'object'

    def neighbor(self, obs: np.ndarray) -> float:
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.threshold

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5

    def inframe(self, obs):
        out = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        keypoints, reversemask = blob_detect(
            out,
            self.min_range,
            self.max_range
        )
        if len(keypoints) == 0:
            return False, reversemask
        return True, reversemask


class MazePosition:
    def __init__(
        self,
        pos: np.ndarray,
        characteristics: Dict[str, Union[int, float, np.ndarray]],
        reward_scale: float = 1.0,
    ) -> None:
        assert 0.0 <= reward_scale <= 1.0
        self.pos = pos
        self.dim = pos.shape[0] - 1
        self.reward_scale = reward_scale
        self.is_target = characteristics['target']
        self.threshold = characteristics['threshold']
        self.position_error_threshold = characteristics['position_error_threshold']
        self.custom_size = None
        self.kind = 'position'
        self.ori = pos[-1]

    def neighbor(self, pos: np.ndarray) -> float:
        return np.linalg.norm(pos[: self.dim] - self.pos[:self.dim]) <= self.threshold

    def euc_dist(self, pos: np.ndarray) -> float:
        return np.sum(np.square(pos[: self.dim] - self.pos[:self.dim])) ** 0.5

    def ori_dist(self, ori: float) -> float:
        return np.abs(ori - self.ori)

    def attained(self, pos: np.ndarray) -> bool:
        return self.euc_dist(pos[:self.dim]) + self.ori_dist(self.pos[-1]) < self.position_error_threshold


class Maze(ABC):
    """Base Class for all maze tasks

    :param structure: List of lists describing the structure of the maze.
    :type structure: List[List[int]]
    :param objects: List of objects to be placed in the maze.
    :type objects: List[MazeObject]
    :param goal_index: Index of target object in `objects`.
    :type goal_index: int
    :param scale: Reward Scaling Factor.
    :type scale: float
    """
    def __init__(
            self,
            structure: List[List[int]],
            objects: List[Union[MazeObject, MazePosition]],
            goal_index: int,
            scale: float,
            reward_threshold: float
    ) -> None:
        self.structure = structure
        self.objects = objects
        self.goal_index = goal_index
        self.scale = scale
        self.reward_threshold = reward_threshold

    def reset(
            self,
            structure: List[List[int]],
            objects: List[MazeObject],
            goal_index: int
    ):
        """Reset method for the maze.

        :param structure: List of lists describing the structure of the maze.
        :type structure: List[List[int]]
        :param objects: List of objects to be placed in the maze.
        :type objects: List[MazeObject]
        :param goal_index: Index of target object in `objects`.
        :type goal_index: int
        :return: Return Status.
        :rtype: bool
        """
        self.structure = structure
        self.objects = objects
        self.goal_index = goal_index

    @abstractmethod
    def reward(
            self,
            observations: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> float:
        """Method to compute reward given current observations from the environment.
        Need to appropriately decide on the input parameters for the method.
        """
        raise NotImplementedError

    @abstractmethod
    def termination(
            self,
            observations: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> bool:
        """Method to check if the current observations imply end of episode.

        :param observations: Current Observations
        :type observations: Union[np.ndarray, Dict[str, np.ndarray]]
        """
        raise NotImplementedError


MAPS = {
        'u_maze': [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ],
        'simple_room': [
            [B, B, B, B, B],
            [B, E, E, E, B],
            [B, R, E, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B]
        ],
        'square_room': [
            [B, B, B, B, B],
            [B, E, E, E, B],
            [B, E, R, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ],
        'u_maze_push': [
            [B, B, B, B, B],
            [B, E, R, B, B],
            [B, E, M, E, B],
            [B, B, E, B, B],
            [B, B, B, B, B],
        ],
        'u_maze_fall': [
            [B, B, B, B],
            [B, R, E, B],
            [B, E, M, B],
            [B, C, C, B],
            [B, E, E, B],
            [B, B, B, B],
        ],
        'local_planner': [
            [E, E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E, E],
            [E, E, E, E, R, E, E, E, E],
            [E, E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E, E],
        ]
}


def create_simple_room_maze(
        maze_size_scaling: float = 4.0,
):
    structure = MAPS['simple_room']
    num_objects = 2
    torso_init = find_robot(structure, maze_size_scaling)
    objects = []
    _open_position_indices = []
    agent_pos = None
    for i in range(len(structure)):
        for j in range(len(structure[i])):
            if not structure[i][j].is_wall_or_chasm():
                _open_position_indices.append([i, j])
            if structure[i][j].is_robot():
                agent_pos = [i, j]
    # print(_open_position_indices)

    _eligible_position_indices = []
    for pos in _open_position_indices:
        # print(agent_pos, pos)
        if not check_target_object_distance(agent_pos, pos):
            _eligible_position_indices.append(pos)

    assert agent_pos is not None
    # print(num_objects)
    # print(_eligible_position_indices)
    object_structure_indices = random.sample(_eligible_position_indices, num_objects)
    # print(object_structure_indices)
    goal_index = np.random.randint(low=0, high=num_objects)

    available_hsv = [
            colorsys.rgb_to_hsv(*rgb) for rgb in params['available_rgb']
    ]
    available_shapes = params['available_shapes']
    target_shape = params['target_shape']
    target_hsv = colorsys.rgb_to_hsv(*params['target_rgb'])
    target_rgb = params['target_rgb']
    objects = []

    for i, [row, col] in enumerate(object_structure_indices):
        site_type = random.choice(available_shapes)
        r, g, b = 0, 0, 0
        h, s, v = 0, 0, 0
        rgb = None
        if goal_index == i:
            rgb = Rgb(*target_rgb)
            h, s, v = copy.deepcopy(target_hsv)
            site_type = target_shape
        else:
            object_hsv = random.choice(available_hsv)
            h, s, v = object_hsv
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            rgb = Rgb(r, g, b)
        size = maze_size_scaling * 0.25
        if site_type != 'sphere':
            size = np.random.uniform(
                    low=size / 3,
                    high=size,
                    size=(3,)).tolist()
        else:
            size = [size, size, size]
        hsv_low = []
        hsv_high = []
        if h > 10 / 180:
            hsv_low.append(h * 180 - 10)
        else:
            hsv_low.append(0)
        if h < 160 / 180:
            hsv_high.append(h * 180 + 10)
        else:
            hsv_high.append(180)
        if s > 100 / 255:
            hsv_low.append(s * 255 - 100)
        else:
            hsv_low.append(0)
        if s < 155 / 255:
            hsv_high.append(s * 255 + 100)
        else:
            hsv_high.append(255)
        hsv_low.append(0)
        hsv_high.append(255)
        objects.append(MazeObject(
                pos=np.array([
                        col * maze_size_scaling - torso_init[0],
                        row * maze_size_scaling - torso_init[1]
                        ], dtype=np.float32),
                characteristics={
                        'hsv_low': copy.deepcopy(hsv_low),
                        'hsv_high': copy.deepcopy(hsv_high),
                        'threshold': 1.5 if i == goal_index else 1.5,
                        'target': True if i == goal_index else False,
                        'rgb': copy.deepcopy(rgb),
                        'size': copy.deepcopy(size),
                        'site_type': site_type
                    })
                )

    maze = SimpleRoom(
        structure=structure,
        objects=objects,
        goal_index=goal_index,
        scale=1.0,
        reward_threshold=1.0
    )
    return maze, structure, _open_position_indices, agent_pos


class SimpleRoom(Maze):
    def __init__(
            self,
            structure: List[List[MazeCell]],
            objects: List[MazeObject],
            goal_index: int,
            scale: float,
            reward_threshold: float
    ) -> None:
        super().__init__(structure, objects, goal_index, scale, reward_threshold)

    def reward(self, observations: Union[np.ndarray, Dict[str, np.ndarray]]) -> float:
        reward = 0.0
        for i, obj in enumerate(self.objects):
            if i == self.goal_index:
                if observations['inframe']:
                    reward += 0.5 * obj.reward_scale + (
                            1 - np.linalg.norm(
                                    observations['pos'] - obj.pos
                            ) / np.linalg.norm(
                                    observations['start_pos'] - obj.pos))
                    if np.linalg.norm(observations['pos'] - obj.pos) <= 2.5 * obj.threshold:
                        reward += 1.0 * obj.reward_scale
            else:
                if obj.neighbor(observations['pos']):
                    reward += -0.1
        return reward

    def termination(self, observations: Union[np.ndarray, Dict[str, np.ndarray]]) -> bool:
        if self.objects[self.goal_index].neighbor(observations['pos']):
            return True
        return False


def create_local_planner_area(
        maze_size_scaling: float = 4.0,
):
    structure = MAPS['local_planner']
    num_objects = 1
    torso_init = find_robot(structure, maze_size_scaling)
    objects = []
    _open_position_indices = []
    agent_pos = None
    for i in range(len(structure)):
        for j in range(len(structure[i])):
            if not structure[i][j].is_wall_or_chasm():
                _open_position_indices.append([i, j])
            if structure[i][j].is_robot():
                agent_pos = [i, j]
    # print(_open_position_indices)

    _eligible_position_indices = []
    for pos in _open_position_indices:
        # print(agent_pos, pos)
        if not check_target_object_distance(agent_pos, pos):
            _eligible_position_indices.append(pos)

    assert agent_pos is not None
    # print(num_objects)
    # print(_eligible_position_indices)
    object_structure_indices = random.sample(_eligible_position_indices, num_objects)
    # print(object_structure_indices)
    goal_index = np.random.randint(low=0, high=num_objects)

    objects = []

    for i, [row, col] in enumerate(object_structure_indices):
        ori = np.random.uniform(low=-np.pi, high=np.pi)
        objects.append(MazePosition(
                pos=np.array([
                        col * maze_size_scaling - torso_init[0],
                        row * maze_size_scaling - torso_init[1],
                        ori
                        ], dtype=np.float32),
                characteristics={
                        'threshold': 1.5 if i == goal_index else 1.5,
                        'target': True if i == goal_index else False,
                        'position_error_threshold' : 1e-2
                    })
                )

    maze = LocalPlanner(
        structure=structure,
        objects=objects,
        goal_index=goal_index,
        scale=1.0,
        reward_threshold=1.0
    )
    return maze, structure, _open_position_indices, agent_pos


class LocalPlanner(Maze):
    def __init__(
            self,
            structure: List[List[MazeCell]],
            objects: List[MazePosition],
            goal_index: int,
            scale: float,
            reward_threshold: float
    ) -> None:
        super().__init__(structure, objects, goal_index, scale, reward_threshold)

    def reward(self, observations: Union[np.ndarray, Dict[str, np.ndarray]]) -> float:
        euc_dist = self.objects[self.goal_index].euc_dist(observations['pos'][:2])
        ori_dist = self.objects[self.goal_index].ori_dist(observations['pos'][-1])
        return -(euc_dist + ori_dist)

    def termination(self, observations: Union[np.ndarray, Dict[str, np.ndarray]]) -> bool:
        # Need to ensure observations['pos'] contains the appropriate values.
        return self.objects[self.goal_index].attained(observations['pos'])


class TaskRegistry:
    REGISTRY: Dict[str, List[Callable]] = {
        "SimpleRoom": [create_simple_room_maze],
        "SquareRoom": [],
        "BigRoom": [],
        "Maze": []
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[Maze]]:
        return TaskRegistry.REGISTRY[key]
