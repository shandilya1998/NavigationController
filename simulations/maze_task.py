"""Maze tasks that are defined by their map, termination condition, and goals.
"""
import cv2
from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Tuple, Type, Union
import numpy as np
from numpy.core.defchararray import lower
from neurorobotics.utils.cv_utils import blob_detect
from neurorobotics.simulations.maze_env_utils import MazeCell
import copy

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
            objects: List[MazeObject],
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


class Scaling(NamedTuple):
    ant: Optional[float]
    point: Optional[float]
    swimmer: Optional[float]


MAPS = {
        'u_maze': [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ],
        'simple_room': [
            [B, B, B, B, B, B],
            [B, E, E, E, E, B],
            [B, R, E, E, E, B],
            [B, E, E, E, E, B],
            [B, B, B, B, B, B]
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
        ]
}


def create_simple_room_maze():
    structure = MAPS['simple_room']
    objects = []
    _open_position_indices = []
    for i in range(len(structure)):
        for j in range(len(structure[i])):
            if not structure[i][j].is_wall_or_chasm():
                _open_position_indices.append([i, j])
    object_structure_indices = random.sample(_open_position_indices, 2)
    


    goal_index = np.random.randint(low=0, high=len(objects))
    maze = SimpleRoom(
        structure=structure,
        objects=objects,
        goal_index=goal_index,
        scale=1.0,
        reward_threshold=1.0
    )
    return maze


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
        




class GoalReward4Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 4.0)

    def __init__(self, scale: float, goal: Tuple[int, int] = (6.0, -6.0)) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale)]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, B, B, B, B, B, B],
            [B, E, E, E, E, E, E, R, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        ]

class CustomGoalReward4Rooms(GoalReward4Rooms):
    def __init__(self,
        scale: float,
        goal: Tuple[int, int] = (3.0, 4.0),
        ) -> None:
        super().__init__(scale, goal)

    def set(self, goals, torso_init):
        self.goals = []
        self.colors = []
        self.scales = []
        for i, goal in enumerate(goals):
            row, col, characteristics = goal
            if characteristics['target']:
                self.goal_index = i
            self.colors.append(characteristics['rgb'])
            self.scales.append(1.0)
            self.goals.append(MazeVisualGoal(
                np.array([
                    col * self.scale - torso_init[1],
                    row * self.scale - torso_init[0]
                ]), characteristics, self.scales[i]
            ))

    def reward(self, pos: np.ndarray, inframe: bool, start_pos: np.ndarray) -> float:
        reward = 0.0
        for i, goal in enumerate(self.goals):
            if i == self.goal_index:
                if inframe:
                    reward += 0.5 * goal.reward_scale + (1 - np.linalg.norm(pos - goal.pos) / np.linalg.norm(start_pos - goal.pos))
                    if np.linalg.norm(pos - goal.pos) <= 2.5 * goal.threshold:
                        reward += 1.0 * goal.reward_scale
            else:
                if goal.neighbor(pos):
                    reward += -0.1
        return reward

    def termination(self, pos: np.ndarray, inframe: bool) -> bool:
        if self.goals[self.goal_index].neighbor(pos):
            return True
        return False

    @staticmethod
    def create_simple_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, R, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        ]


class CustomGoalReward4RoomsV2(GoalReward4Rooms):
    def __init__(self,
        scale: float,
        goal: Tuple[int, int] = (6, -6),
        ) -> None:
        super().__init__(scale, goal)
        self.set()

    def set(self, offset = 0.2):
        self.goal_index = np.random.randint(low = 0, high = 4)
        self.colors = []
        self.scales = []
        self.goals = []
        for i in range(4):
            if i == self.goal_index:
                self.colors.append(copy.deepcopy(RED))
                self.scales.append(1.0)
            else:
                self.colors.append(copy.deepcopy(GREEN))
                self.scales.append(1.0)

        self.goals = [ 
            MazeVisualGoal(np.array([
                np.random.uniform(6.0 - offset, 6.0 + offset) * self.scale,
                -np.random.uniform(6.0 - offset, 6.0 + offset) * self.scale
            ]), self.scales[0], self.colors[0], 2.25),
            MazeVisualGoal(np.array([
                np.random.uniform(6.0 - offset, 6.0 + offset) * self.scale,
                -np.random.uniform(-6.0 - offset, -6.0 + offset) * self.scale
            ]), self.scales[1], self.colors[1], 2.25),
            MazeVisualGoal(np.array([
                np.random.uniform(-6.0 - offset, -6.0 + offset) * self.scale,
                -np.random.uniform(6.0 - offset, 6.0 + offset) * self.scale 
            ]), self.scales[2], self.colors[2], 2.25),
            MazeVisualGoal(np.array([
                np.random.uniform(-6.0 - offset, -6.0 + offset) * self.scale,
                -np.random.uniform(-6.0 - offset, -6.0 + offset) * self.scale 
            ]), self.scales[3], self.colors[3], 2.25),
        ]   

    def reward(self, pos: np.ndarray, inframe: bool) -> float:
        goal = self.goals[self.goal_index]
        reward = 0.0 
        if inframe:
            reward = 0.5 * goal.reward_scale
        if np.linalg.norm(pos - goal.pos) <= 2.5 * goal.threshold:
            reward += 1.0 * goal.reward_scale
        return reward

    def termination(self, pos: np.ndarray, inframe: bool) -> bool:
        if self.goals[self.goal_index].neighbor(pos):
            return True
        return False


    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, B, B, E, B, B, B, B, B, E, B, B, B, B, B, E, B, B, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, R, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, B, B, E, B, B, B, B, B, E, B, B, B, B, B, E, B, B, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        ]


class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "SimpleRoom": [GoalRewardSimple, DistRewardSimpleRoom, GoalRewardSimpleRoom],
        "SquareRoom": [DistRewardSquareRoom, GoalRewardSquareRoom, NoRewardSquareRoom],
        "UMaze": [DistRewardUMaze, GoalRewardUMaze],
        "Push": [DistRewardPush, GoalRewardPush],
        "Fall": [DistRewardFall, GoalRewardFall],
        "2Rooms": [DistReward2Rooms, GoalReward2Rooms, SubGoal2Rooms],
        "4Rooms": [DistReward4Rooms, GoalReward4Rooms, CustomGoalReward4Rooms, GoalRewardNoObstacle],
        "TRoom": [DistRewardTRoom, GoalRewardTRoom, SubGoalTRoom],
        "BlockMaze": [DistRewardBlockMaze, GoalRewardBlockMaze],
        "Corridor": [DistRewardCorridor, GoalRewardCorridor, NoRewardCorridor],
        "Billiard": [
            DistRewardBilliard,  # v0
            GoalRewardBilliard,  # v1
            SubGoalBilliard,  # v2
            BanditBilliard,  # v3
            NoRewardBilliard,  # v4
        ],
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return TaskRegistry.REGISTRY[key]
